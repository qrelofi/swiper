import argparse
import json
import os
import time
import random
import logging
import unicodedata
import sqlite3
import re
from playwright.sync_api import sync_playwright
import urllib.parse
import subprocess
import pty
import errno
import sys
from typing import Dict, List
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import threading
import uuid
import signal
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes
import asyncio
from dotenv import load_dotenv
try:
    from playwright_stealth import stealth_sync
except Exception:
    stealth_sync = lambda page: None
from instagrapi import Client
from instagrapi.exceptions import ChallengeRequired, TwoFactorRequired, PleaseWaitFewMinutes, RateLimitError, LoginRequired
import psutil
from queue import Queue, Empty


load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_bot.log'),
        logging.StreamHandler()
    ]
)

user_fetching = set()
user_cancel_fetch = set()  # new set
AUTHORIZED_FILE = 'authorized_users.json'
TASKS_FILE = 'tasks.json'
_owner = os.environ.get('OWNER_TG_ID', '')
try:
    OWNER_TG_ID = int(_owner) if _owner and _owner.strip() else 0
except ValueError:
    logging.warning("Invalid OWNER_TG_ID '%s', defaulting to 0", _owner)
    OWNER_TG_ID = 0

BOT_TOKEN = os.environ.get('BOT_TOKEN')
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

authorized_users = []  # list of {'id': int, 'username': str}
users_data: Dict[int, Dict] = {}  # unlocked data {'accounts': list, 'default': int, 'pairs': dict or None, 'switch_minutes': int, 'threads': int}
users_pending: Dict[int, Dict] = {}  # pending challenges
users_tasks: Dict[int, List[Dict]] = {}  # tasks per user
persistent_tasks = []
running_processes: Dict[int, subprocess.Popen] = {}
waiting_for_otp = {}
user_queues = {}
user_fetching = set()

# Ensure sessions directory exists
os.makedirs('sessions', exist_ok=True)

# Helper: locate the sender script (swiper.py) with backwards compatibility
def _find_sender_script() -> str:
    base_dir = os.path.dirname(__file__)
    candidates = ['swiper.py', 'sesion msg.py']
    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    # default path (may not exist yet)
    return os.path.join(base_dir, 'swiper.py')

# === PATCH: Fix instagrapi invalid timestamp bug ===
def _sanitize_timestamps(obj):
    """Fix invalid *_timestamp_us fields in Instagram data"""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, int) and k.endswith("_timestamp_us"):
                try:
                    secs = int(v) // 1_000_000  # convert microseconds → seconds
                except Exception:
                    secs = None
                # skip impossible years (>2100 or negative)
                if secs is None or secs < 0 or secs > 4102444800:
                    new_obj[k] = None
                else:
                    new_obj[k] = secs
            else:
                new_obj[k] = _sanitize_timestamps(v)
        return new_obj
    elif isinstance(obj, list):
        return [_sanitize_timestamps(i) for i in obj]
    else:
        return obj


async def playwright_login_and_save_state(username: str, password: str, user_id: int) -> str:
    """
    Async Playwright login.
    - Instagram me login karta hai
    - storage_state ko sessions/<user>_<username>_state.json me save karta hai
    - file path return karta hai
    """
    COOKIE_FILE = f"sessions/{user_id}_{username}_state.json"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = await browser.new_context(
            user_agent=USER_AGENT,  # tumhara existing USER_AGENT
            viewport={"width": 1280, "height": 720},
        )

        page = await context.new_page()

        login_url = "https://www.instagram.com/accounts/login/?__coig_login=1"
        logging.info("[PLOGIN-ASYNC] Goto %s", login_url)

        await page.goto(login_url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        logging.info(
            "[PLOGIN-ASYNC] After goto -> URL=%s TITLE=%s",
            page.url,
            await page.title(),
        )

        # ---------- CHECK LOGIN FORM ----------
        username_inputs = await page.locator('input[name="username"]').count()
        if username_inputs == 0:
            logging.warning(
                "[PLOGIN-ASYNC] Username field not found. URL=%s",
                page.url,
            )
            await page.wait_for_timeout(5000)
            username_inputs = await page.locator('input[name=\"username\"]').count()

        if username_inputs == 0:
            # Still nahi mila → intro/splash
            html_snippet = (await page.content())[:1000].replace("\n", " ")
            logging.warning(
                "[PLOGIN-ASYNC] Login form NOT loaded. URL=%s SNIPPET=%s",
                page.url,
                html_snippet,
            )
            await browser.close()
            raise ValueError("ERROR_010: Instagram login form not loaded (stuck on intro/splash)")

        # ---------- HUMAN-LIKE LOGIN ----------
        username_input = page.locator('input[name="username"]')
        password_input = page.locator('input[name="password"]')
        login_button = page.locator('button[type="submit"]').first

        # Username typing
        await username_input.click()
        await page.wait_for_timeout(random.randint(300, 900))
        await username_input.fill("")  # clear
        await username_input.type(username, delay=random.randint(60, 140))  # ms per char

        # Password typing
        await page.wait_for_timeout(random.randint(300, 900))
        await password_input.click()
        await page.wait_for_timeout(random.randint(200, 700))
        await password_input.fill("")
        await password_input.type(password, delay=random.randint(60, 140))

        # Click login with tiny jitter
        await page.wait_for_timeout(random.randint(400, 1000))
        await login_button.click()
        logging.info("[PLOGIN-ASYNC] Submitted login form for %s", username)

        # ---------- POST-LOGIN CHECK (OTP / SUCCESS) ----------
        # Thoda time do redirect ke liye
        await page.wait_for_timeout(5000)

        current_url = page.url
        logging.info("[PLOGIN-ASYNC] After login URL=%s", current_url)

        # Quick OTP detection without long timeout
        otp_locator = page.locator('input[name="verificationCode"]')
        otp_count = await otp_locator.count()

        if otp_count > 0 or "challenge" in current_url or "two_factor" in current_url:
            logging.info(
                "[PLOGIN-ASYNC] OTP / challenge detected for user %s (url=%s, otp_count=%s)",
                username,
                current_url,
                otp_count,
            )
            await browser.close()
            # Abhi ke liye clear error; baad me Telegram OTP flow hook kar sakte hain
            raise ValueError("ERROR_OTP: OTP / challenge required. Please use session/2FA flow.")

        # Agar yahan tak aa gaye aur URL jaise:
        # - /accounts/onetap/...
        # - / (home feed, profile, etc.)
        # to login successful maan lo
        logging.info("[PLOGIN-ASYNC] No OTP required, login looks successful. URL=%s", current_url)

        # Kuch extra wait, phir state save
        await page.wait_for_timeout(4000)

        await context.storage_state(path=COOKIE_FILE)
        logging.info("[PLOGIN-ASYNC] storage_state saved to %s", COOKIE_FILE)

        await browser.close()
        logging.info("[PLOGIN-ASYNC] Browser closed for user_id=%s", user_id)

    return COOKIE_FILE


# 🧩 Monkeypatch instagrapi to fix validation crash
try:
    import instagrapi.extractors as extractors
    _orig_extract_reply_message = extractors.extract_reply_message

    def patched_extract_reply_message(data):
        data = _sanitize_timestamps(data)
        return _orig_extract_reply_message(data)

    extractors.extract_reply_message = patched_extract_reply_message
    print("[Patch] Applied timestamp sanitizer to instagrapi extractors ✅")
except Exception as e:
    print(f"[Patch Warning] Could not patch instagrapi: {e}")
# === END PATCH ===

# --- Playwright sync helper: run sync_playwright() inside a fresh thread ---
def run_with_sync_playwright(fn, *args, **kwargs):
    """
    Runs `fn(p, *args, **kwargs)` where p is the object returned by sync_playwright()
    inside a new thread and returns fn's return value (or raises exception).
    """
    result = {"value": None, "exc": None}

    def target():
        try:
            with sync_playwright() as p:
                result["value"] = fn(p, *args, **kwargs)
        except Exception as e:
            result["exc"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if result["exc"]:
        raise result["exc"]
    return result["value"]

def load_authorized():
    global authorized_users
    if os.path.exists(AUTHORIZED_FILE):
        with open(AUTHORIZED_FILE, 'r') as f:
            authorized_users = json.load(f)
    # Ensure owner is authorized
    if not any(u['id'] == OWNER_TG_ID for u in authorized_users):
        authorized_users.append({'id': OWNER_TG_ID, 'username': 'owner'})

load_authorized()

def load_users_data():
    global users_data
    users_data = {}
    for file in os.listdir('.'):
        if file.startswith('user_') and file.endswith('.json'):
            user_id_str = file[5:-5]
            if user_id_str.isdigit():
                user_id = int(user_id_str)
                with open(file, 'r') as f:
                    data = json.load(f)
                # Defaults
                if 'pairs' not in data:
                    data['pairs'] = None
                if 'switch_minutes' not in data:
                    data['switch_minutes'] = 10
                if 'threads' not in data:
                    data['threads'] = 1
                users_data[user_id] = data

load_users_data()

def save_authorized():
    with open(AUTHORIZED_FILE, 'w') as f:
        json.dump(authorized_users, f)

def save_user_data(user_id: int, data: Dict):
    with open(f'user_{user_id}.json', 'w') as f:
        json.dump(data, f)

def is_authorized(user_id: int) -> bool:
    return any(u['id'] == user_id for u in authorized_users)

def is_owner(user_id: int) -> bool:
    return user_id == OWNER_TG_ID

def future_expiry(days=365):
    return int(time.time()) + days*24*3600

def convert_for_playwright(insta_file, playwright_file):
    try:
        with open(insta_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        return

    cookies = []
    auth = data.get("authorization_data", {})
    for name, value in auth.items():
        cookies.append({
            "name": name,
            "value": urllib.parse.unquote(value),
            "domain": ".instagram.com",
            "path": "/",
            "expires": future_expiry(),
            "httpOnly": True,
            "secure": True,
            "sameSite": "Lax"
        })

    playwright_state = {
        "cookies": cookies,
        "origins": [{"origin": "https://www.instagram.com", "localStorage": []}]
    }

    with open(playwright_file, "w") as f:
        json.dump(playwright_state, f, indent=4)

def get_storage_state_from_instagrapi(settings: Dict):
    """Convert instagrapi settings to Playwright storage state.
    Falls back to minimal state if settings are incomplete."""
    cookies = []
    
    # Try to set settings and extract cookies
    try:
        cl = Client()
        cl.set_settings(settings)

        # Collect cookies from instagrapi structures (compatible with multiple instagrapi versions)
        cookies_dict = {}
        if hasattr(cl, "session") and cl.session:
            try:
                cookies_dict = cl.session.cookies.get_dict()
            except Exception:
                cookies_dict = {}
        elif hasattr(cl, "private") and hasattr(cl.private, "cookies"):
            try:
                cookies_dict = cl.private.cookies.get_dict()
            except Exception:
                cookies_dict = {}
        elif hasattr(cl, "_http") and hasattr(cl._http, "cookies"):
            try:
                cookies_dict = cl._http.cookies.get_dict()
            except Exception:
                cookies_dict = {}

        for name, value in cookies_dict.items():
            cookies.append({
                "name": name,
                "value": value,
                "domain": ".instagram.com",
                "path": "/",
                "expires": int(time.time()) + 365*24*3600,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax"
            })
    except Exception as e:
        # If set_settings fails, try to extract cookies from settings dict directly
        try:
            if "cookies" in settings:
                for cookie in settings["cookies"]:
                    if isinstance(cookie, dict):
                        cookies.append(cookie)
        except Exception:
            pass

    storage_state = {
        "cookies": cookies,
        "origins": [{"origin": "https://www.instagram.com", "localStorage": []}]
    }
    return storage_state

def instagrapi_login(username, password):
    cl = Client()
    session_file = f"{username}_session.json"
    playwright_file = f"{username}_state.json"
    try:
        cl.login(username, password)
        cl.dump_settings(session_file)
        convert_for_playwright(session_file, playwright_file)
    except (ChallengeRequired, TwoFactorRequired):
        raise ValueError("ERROR_004: Login challenge or 2FA required")
    except (PleaseWaitFewMinutes, RateLimitError):
        raise ValueError("ERROR_002: Rate limit exceeded")
    except Exception as e:
        raise ValueError(f"ERROR_007: Login failed - {str(e)}")
    return json.load(open(playwright_file))

def list_group_chats(user_id, storage_state, username, password, max_groups=10, amount=10):
    username = username.strip().lower()
    norm_username = username.strip().lower()
    session_file = f"sessions/{user_id}_{norm_username}_session.json"
    playwright_file = f"sessions/{user_id}_{norm_username}_state.json"
    cl = Client()
    updated = False
    new_state = None

    # Load existing session if available
    if os.path.exists(session_file):
        try:
            cl.load_settings(session_file)
        except Exception:
            pass

    try:
        threads = cl.direct_threads(amount=amount)
        time.sleep(random.uniform(1.0, 3.0))
    except LoginRequired:
        cl.login(username, password)
        cl.dump_settings(session_file)
        convert_for_playwright(session_file, playwright_file)
        updated = True
        threads = cl.direct_threads(amount=amount)
        time.sleep(random.uniform(1.0, 3.0))

    groups = []
    for thread in threads:
        if len(groups) >= max_groups:
            break
        if getattr(thread, "is_group", False):
            member_count = len(getattr(thread, "users", [])) + 1
            if member_count < 3:
                continue

            title = getattr(thread, "thread_title", None) or getattr(thread, "title", None)
            if not title or title.strip() == "":
                try:
                    users_part = ", ".join([u.username for u in getattr(thread, "users", [])][:3])
                    display = users_part if users_part else "<no name>"
                except Exception:
                    display = "<no name>"
            else:
                display = title

            url = f"https://www.instagram.com/direct/t/{getattr(thread, 'thread_id', getattr(thread, 'id', 'unknown'))}"
            groups.append({'display': display, 'url': url})

    if updated and os.path.exists(playwright_file):
        new_state = get_storage_state_from_instagrapi(cl.get_settings())
        with open(playwright_file, 'w') as f:
            json.dump(new_state, f)
    elif os.path.exists(playwright_file):
        new_state = json.load(open(playwright_file))
    else:
        new_state = storage_state

    return groups, new_state

def get_dm_thread_url(user_id, username, password, target_username):
    norm_username = username.strip().lower()
    session_file = f"sessions/{user_id}_{norm_username}_session.json"
    playwright_file = f"sessions/{user_id}_{norm_username}_state.json"
    cl = Client()
    updated = False

    if os.path.exists(session_file):
        try:
            cl.load_settings(session_file)
        except Exception:
            pass

    try:
        threads = cl.direct_threads(amount=10)
        time.sleep(random.uniform(1.0, 3.0))
    except LoginRequired:
        cl.login(username, password)
        cl.dump_settings(session_file)
        convert_for_playwright(session_file, playwright_file)
        updated = True
        threads = cl.direct_threads(amount=10)
        time.sleep(random.uniform(1.0, 3.0))

    for thread in threads:
        if not getattr(thread, 'is_group', True) and len(getattr(thread, 'users', [])) == 1:
            try:
                user = thread.users[0]
                if user.username == target_username:
                    thread_id = getattr(thread, 'thread_id', getattr(thread, 'id', None))
                    if thread_id:
                        url = f"https://www.instagram.com/direct/t/{thread_id}/"
                        if updated:
                            settings = cl.get_settings()
                            new_state = get_storage_state_from_instagrapi(settings)
                            with open(playwright_file, 'w') as f:
                                json.dump(new_state, f)
                        return url
            except Exception:
                continue

    return None

def perform_login(page, username, password):
    try:
        page.evaluate("""() => {
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            window.chrome = { app: {}, runtime: {} };
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: 'denied' }) :
                originalQuery(parameters)
            );
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Google Inc. (Intel)';
                if (parameter === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics 630 (0x00003E9B) Direct3D11 vs_5_0 ps_5_0, D3D11)';
                return getParameter.call(this, parameter);
            };
        }""")

        username_locator = page.locator('input[name="username"]')
        username_locator.wait_for(state='visible', timeout=10000)
        username_locator.focus()
        time.sleep(random.uniform(0.5, 1.5))
        for char in username:
            username_locator.press(char)
            time.sleep(random.uniform(0.05, 0.15))

        password_locator = page.locator('input[name="password"]')
        password_locator.wait_for(state='visible', timeout=10000)
        time.sleep(random.uniform(0.5, 1.5))
        password_locator.focus()
        time.sleep(random.uniform(0.3, 0.8))
        for char in password:
            password_locator.press(char)
            time.sleep(random.uniform(0.05, 0.15))

        time.sleep(random.uniform(1.0, 2.5))

        submit_locator = page.locator('button[type="submit"]')
        submit_locator.wait_for(state='visible', timeout=10000)
        if not submit_locator.is_enabled():
            raise Exception("Submit button not enabled")
        submit_locator.click()

        try:
            page.wait_for_url(lambda url: 'accounts/login' not in url and 'challenge' not in url and 'two_factor' not in url, timeout=60000)
            
            if page.locator('[role="alert"]').count() > 0:
                error_text = page.locator('[role="alert"]').inner_text().lower()
                if 'incorrect' in error_text or 'wrong' in error_text:
                    raise ValueError("ERROR_001: Invalid credentials")
                elif 'wait' in error_text or 'few minutes' in error_text or 'too many' in error_text:
                    raise ValueError("ERROR_002: Rate limit exceeded")
                else:
                    raise ValueError(f"ERROR_003: Login error - {error_text}")
        except TimeoutError:
            current_url = page.url
            page_content = page.content().lower()
            if 'challenge' in current_url:
                raise ValueError("ERROR_004: Login challenge required")
            elif 'two_factor' in current_url or 'verify' in current_url:
                raise ValueError("ERROR_005: 2FA verification required")
            elif '429' in page_content or 'rate limit' in page_content or 'too many requests' in page_content:
                raise ValueError("ERROR_002: Rate limit exceeded")
            elif page.locator('[role="alert"]').count() > 0:
                error_text = page.locator('[role="alert"]').inner_text().lower()
                raise ValueError(f"ERROR_006: Login failed - {error_text}")
            else:
                raise ValueError("ERROR_007: Login timeout or unknown error")

        logging.info("Login successful")
    except Exception as e:
        logging.error(f"Login failed: {str(e)}")
        raise

# ---------------- Globals for PTY ----------------
APP = None
LOOP = None
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()

# ---------------- Async Instagram Login ----------------
async def async_instagram_login(user_id: int, username: str, password: str, chat_id: int):
    """Async Instagram login with Telegram OTP handling"""
    cl = Client()
    username = username.strip().lower()
    session_file = f"sessions/{user_id}_{username}_session.json"
    playwright_file = f"sessions/{user_id}_{username}_state.json"

    try:
        # Send initial status
        await APP.bot.send_message(chat_id=chat_id, text=f"🔥[{username}] ⚙️ Attempting login...")

        # Try login
        cl.login(username, password)
        cl.dump_settings(session_file)
        convert_for_playwright(session_file, playwright_file)

        await APP.bot.send_message(chat_id=chat_id, text=f"✅[{username}] Logged in successfully! Session saved.")
        return True

    except TwoFactorRequired:
        # 2FA required - prompt user through Telegram
        await APP.bot.send_message(
            chat_id=chat_id,
            text=f"🔐 [{username}] 2FA Required!\n\n📧 Check your email/SMS for the 6-digit code and reply with it here.\n⏰ You have 2 minutes to enter the code."
        )

        # Set up OTP waiting
        waiting_for_otp[user_id] = True
        user_queues[user_id] = Queue()

        try:
            # Wait for OTP with timeout
            otp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, user_queues[user_id].get),
                timeout=120  # 2 minutes
            )

            # Try to complete login with OTP
            cl.login(username, password, verification_code=otp)
            cl.dump_settings(session_file)
            convert_for_playwright(session_file, playwright_file)

            await APP.bot.send_message(chat_id=chat_id, text=f"✅[{username}] 2FA resolved! Logged in successfully.")
            return True

        except asyncio.TimeoutError:
            await APP.bot.send_message(chat_id=chat_id, text=f"⏰[{username}] OTP timeout! Login cancelled.")
            return False
        except Exception as e:
            await APP.bot.send_message(chat_id=chat_id, text=f"❌[{username}] OTP failed: {str(e)}")
            return False
        finally:
            # Clean up
            waiting_for_otp.pop(user_id, None)
            user_queues.pop(user_id, None)

    except ChallengeRequired:
        # Challenge required - prompt user through Telegram
        await APP.bot.send_message(
            chat_id=chat_id,
            text=f"🛡️ [{username}] Security Challenge Required!\n\n📧 Check your email/SMS for the 6-digit code and reply with it here.\n⏰ You have 2 minutes to enter the code."
        )

        # Set up OTP waiting
        waiting_for_otp[user_id] = True
        user_queues[user_id] = Queue()

        try:
            # Wait for OTP with timeout
            otp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, user_queues[user_id].get),
                timeout=120  # 2 minutes
            )

            # Try to resolve challenge
            cl.challenge_resolve(cl.last_json, security_code=otp)
            cl.dump_settings(session_file)
            convert_for_playwright(session_file, playwright_file)

            await APP.bot.send_message(chat_id=chat_id, text=f"✅[{username}] Challenge resolved! Logged in successfully.")
            return True

        except asyncio.TimeoutError:
            await APP.bot.send_message(chat_id=chat_id, text=f"⏰[{username}] Challenge timeout! Login cancelled.")
            return False
        except Exception as e:
            await APP.bot.send_message(chat_id=chat_id, text=f"❌[{username}] Challenge failed: {str(e)}")
            return False
        finally:
            # Clean up
            waiting_for_otp.pop(user_id, None)
            user_queues.pop(user_id, None)

    except Exception as e:
        error_msg = str(e)
        if "checkpoint_required" in error_msg.lower():
            await APP.bot.send_message(chat_id=chat_id, text=f"🚫[{username}] Instagram checkpoint required. Try again later or use a different account.")
        elif "rate limit" in error_msg.lower():
            await APP.bot.send_message(chat_id=chat_id, text=f"🐌[{username}] Rate limited. Wait a few minutes and try again.")
        else:
            await APP.bot.send_message(chat_id=chat_id, text=f"❌[{username}] Login failed: {error_msg}")
        return False

# ---------------- PTY reader thread ----------------
def reader_thread(user_id: int, chat_id: int, master_fd: int, username: str, password: str):
    global APP, LOOP
    buf = b""
    while True:
        try:
            data = os.read(master_fd, 1024)
            if not data:
                break
            buf += data
            while b"\n" in buf or len(buf) > 2048:
                if b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(errors="ignore").strip()
                else:
                    text = buf.decode(errors="ignore")
                    buf = b""
                if not text:
                    continue
                if text.startswith("Code entered"):
                    continue
                lower = text.lower()
                if (
                    len(text) > 300
                    or "cdninstagram.com" in lower
                    or "http" in lower
                    or "{" in text
                    or "}" in text
                    or "debug" in lower
                    or "info" in lower
                    or "urllib3" in lower
                    or "connection" in lower
                    or "starting new https" in lower
                    or "instagrapi" in lower
                ):
                    continue
                try:
                    if APP and LOOP:
                        asyncio.run_coroutine_threadsafe(
                            APP.bot.send_message(chat_id=chat_id, text=f"🔥{text}"), LOOP
                        )
                except Exception:
                    logging.error("[THREAD] send_message failed")
        except OSError as e:
            if e.errno == errno.EIO:
                break
            else:
                logging.error("[THREAD] PTY read error: %s", e)
                break
        except Exception as e:
            logging.error("[THREAD] Unexpected error: %s", e)
            break
    try:
        playwright_file = f"sessions/{user_id}_{username}_state.json"
        if os.path.exists(playwright_file):
            with open(playwright_file, 'r') as f:
                state = json.load(f)
            if user_id in users_data:
                data = users_data[user_id]
            else:
                data = {'accounts': [], 'default': None, 'pairs': None, 'switch_minutes': 10, 'threads': 1}
            # normalize incoming username
            norm_username = username.strip().lower()

            for i, acc in enumerate(data['accounts']):
                if acc.get('ig_username', '').strip().lower() == norm_username:
                    # overwrite existing entry for exact same username (normalized)
                    data['accounts'][i] = {'ig_username': norm_username, 'password': password, 'storage_state': state}
                    data['default'] = i
                    break
            else:
                # not found -> append new normalized account
                data['accounts'].append({'ig_username': norm_username, 'password': password, 'storage_state': state})
                data['default'] = len(data['accounts']) - 1
            save_user_data(user_id, data)
            users_data[user_id] = data
            if APP and LOOP:
                asyncio.run_coroutine_threadsafe(APP.bot.send_message(chat_id=chat_id, text="✅ Login successful and saved securely! 🎉"), LOOP)
        else:
            if APP and LOOP:
                asyncio.run_coroutine_threadsafe(APP.bot.send_message(chat_id=chat_id, text="⚠️ Login failed. No session saved."), LOOP)
    except Exception as e:
        logging.error("Failed to save user data: %s", e)
        if APP and LOOP:
            asyncio.run_coroutine_threadsafe(APP.bot.send_message(chat_id=chat_id, text=f"⚠️ Error saving data: {str(e)}"), LOOP)
    finally:
        with SESSIONS_LOCK:
            if user_id in SESSIONS:
                try:
                    os.close(SESSIONS[user_id]["master_fd"])
                except Exception:
                    pass
                SESSIONS.pop(user_id, None)

# ---------------- Relay input ----------------
async def relay_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    with SESSIONS_LOCK:
        info = SESSIONS.get(user_id)
    if not info:
        return
    master_fd = info["master_fd"]
    try:
        os.write(master_fd, (text + "\n").encode())
    except OSError as e:
        await update.message.reply_text(f"Failed to write to PTY stdin: {e}")
    except Exception as e:
        logging.error("Relay input error: %s", e)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if user_id in waiting_for_otp:
        if len(text) == 6 and text.isdigit():
            user_queues[user_id].put(text)
            del waiting_for_otp[user_id]
            await update.message.reply_text("✅ Code submitted to browser! 🔄")
            return
        else:
            await update.message.reply_text("❌ Invalid code. Please enter 6-digit code.")
            return
    # Fallback to relay
    await relay_input(update, context)

# ---------------- Kill command ----------------
async def cmd_kill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with SESSIONS_LOCK:
        info = SESSIONS.get(user_id)
    if not info:
        await update.message.reply_text("No active PTY session.")
        return
    pid = info["pid"]
    master_fd = info["master_fd"]
    try:
        os.kill(pid, 15)
    except Exception:
        pass
    try:
        os.close(master_fd)
    except Exception:
        pass
    with SESSIONS_LOCK:
        SESSIONS.pop(user_id, None)
    await update.message.reply_text(f"🛑 Stopped login terminal (pid={pid}) successfully.")

# ---------------- Flush command ----------------
async def flush(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⚠️ you are not an admin ⚠️")
        return
    global users_tasks, persistent_tasks
    for uid, tasks in users_tasks.items():
        for task in tasks[:]:
            proc = task['proc']
            proc.terminate()
            await asyncio.sleep(3)
            if proc.poll() is None:
                proc.kill()
            # remove from runtime map if present
            pid = task.get('pid')
            if pid in running_processes:
                running_processes.pop(pid, None)
            if task.get('type') == 'message_attack' and 'names_file' in task:
                names_file = task['names_file']
                if os.path.exists(names_file):
                    os.remove(names_file)
            logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Task stop user={uid} task={task['id']} by flush")
            mark_task_stopped_persistent(task['id'])
            tasks.remove(task)
        users_tasks[uid] = tasks
    await update.message.reply_text("🛑 All tasks globally stopped! 🛑")

USERNAME, PASSWORD = range(2)
PLO_USERNAME, PLO_PASSWORD = range(2)
SLOG_SESSION, SLOG_USERNAME = range(2)
ENGAGE_TEXT = 0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("<b><i>🌌 ✨ LOFI PAPA — Control Deck ✨ 🌌</i></b>\n<b>— Send vibes. Send messages. —</b>\nType /help to explore features\n\n<b>🎛️ Powered by LOFI</b> <i>— spam mode</i> 💫", parse_mode="HTML")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! lofi ⚠️")
        return
    help_text = """<b>🎛️ INSTAGRAM CONTROL PANEL</b>
<i>High-Speed | Stable | Pro</i>

<b>➤ AUTH & LOGIN</b>
──────────────────────────────────────
/login        📱  Standard Instagram login
/plogin       🔐  Playwright human-like login
/slogin       🔑  Login using session string
/logout &lt;id&gt;  🚪  Remove logged account
/kill         🛑  Terminate active login process

<b>➤ ACCOUNT CONTROL</b>
──────────────────────────────────────
/viewmyac     👀  Show all saved accounts
/setig &lt;no&gt;   🔄  Select default account
/pair ig1-ig2 📦  Pair accounts for rotation
/unpair       ✨  Remove pairing (single/all)

<b>➤ PERFORMANCE SETTINGS</b>
──────────────────────────────────────
/switch &lt;min&gt; ⏱️  Account switch interval (min 5)
/threads &lt;1-5&gt; 🔢 Message threads control
/viewpref    ⚙️  View current configuration

<b>➤ MESSAGE ENGINE</b>
──────────────────────────────────────
/attack      💥  Start message delivery
/stop &lt;id/all&gt; 🛑 Stop running task(s)
/task        📋  View active processes

<b>➤ SYSTEM STATUS</b>
──────────────────────────────────────
/usg         📊  CPU / RAM / Load usage

"""
    if is_owner(user_id):
        help_text += """<b>🔸 Admin</b>
/ add ➕ &lt;tg_id&gt; - Add authorized user
/ remove ➖ &lt;tg_id&gt; - Remove authorized user
/ users 📜 - List authorized users
/ flush 🧹 - Stop all tasks globally
"""
    help_text += "\n<b>💫 Enjoy the flow — Keep it smooth 💫</b>"
    await update.message.reply_text(help_text, parse_mode="HTML")

async def login_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return ConversationHandler.END
    await update.message.reply_text("📱 Enter Instagram username: 📱")
    return USERNAME

async def get_username(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Normalize username: remove surrounding spaces and lowercase
    context.user_data['ig_username'] = update.message.text.strip().lower()
    await update.message.reply_text("🔒 Enter password: 🔒")
    return PASSWORD

async def get_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = context.user_data['ig_username']
    password = update.message.text.strip()

    # Check if already logging in
    if user_id in user_fetching:
        await update.message.reply_text("⚠️ Login already in progress. Please wait.")
        return ConversationHandler.END

    user_fetching.add(user_id)

    try:
        # Use async login instead of PTY
        success = await async_instagram_login(user_id, username, password, chat_id)

        if success:
            # Save account data
            playwright_file = f"sessions/{user_id}_{username}_state.json"
            if os.path.exists(playwright_file):
                with open(playwright_file, 'r') as f:
                    state = json.load(f)

                if user_id in users_data:
                    data = users_data[user_id]
                else:
                    data = {'accounts': [], 'default': None, 'pairs': None, 'switch_minutes': 10, 'threads': 1}

                # Normalize username
                norm_username = username.strip().lower()

                # Check if account already exists
                for i, acc in enumerate(data['accounts']):
                    if acc.get('ig_username', '').strip().lower() == norm_username:
                        # Update existing
                        data['accounts'][i] = {'ig_username': norm_username, 'password': password, 'storage_state': state}
                        data['default'] = i
                        break
                else:
                    # Add new
                    data['accounts'].append({'ig_username': norm_username, 'password': password, 'storage_state': state})
                    data['default'] = len(data['accounts']) - 1

                save_user_data(user_id, data)
                users_data[user_id] = data

                await update.message.reply_text("✅ Login successful and saved securely! 🎉")
            else:
                await update.message.reply_text("⚠️ Login completed but session file not found.")
        else:
            await update.message.reply_text("❌ Login failed. Check credentials and try again.")

    except Exception as e:
        logging.error(f"Login error for user {user_id}: {e}")
        await update.message.reply_text(f"⚠️ Login error: {str(e)}")
    finally:
        user_fetching.discard(user_id)

    return ConversationHandler.END

# --- /plogin handlers (ASYNC, NO THREAD) ---
async def plogin_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return ConversationHandler.END

    await update.message.reply_text("🔐 Enter Instagram username for Playwright login: ")
    return PLO_USERNAME


async def plogin_get_username(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['pl_username'] = update.message.text.strip().lower()
    await update.message.reply_text("🔒 Enter password: 🔒")
    return PLO_PASSWORD


async def plogin_get_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = context.user_data['pl_username']
    password = update.message.text.strip()

    await update.message.reply_text("🔄 Starting Playwright login... (async, no threads)")

    try:
        # 1) Playwright se login + storage_state save
        state_file = await playwright_login_and_save_state(username, password, user_id)

        # 2) JSON state load karke Instagrapi me convert karo (tumhara purana logic reuse)
        logging.info("[PLOGIN] Loading storage_state from %s", state_file)
        return ConversationHandler.END
        state = json.load(open(state_file))

        cookies = [c for c in state.get('cookies', []) if c.get('domain') == '.instagram.com']
        logging.info("[PLOGIN] cookies for .instagram.com = %s", len(cookies))

        # Extract sessionid cookie (Playwright → Instagrapi)
        sessionid = None
        for c in cookies:
            if c.get("name") == "sessionid":
                sessionid = c.get("value")
                break

        if not sessionid:
            logging.error("[PLOGIN] sessionid cookie not found in storage_state")
            raise ValueError("ERROR_011: sessionid cookie not found – cannot init Instagrapi client")

        cl = Client()
        logging.info("[PLOGIN] Logging into Instagrapi using sessionid (len=%s)", len(sessionid))

        # Instagrapi new valid method
        cl.login_by_sessionid(sessionid)

        session_file = f"sessions/{user_id}_{username}_session.json"
        logging.info("[PLOGIN] Dumping Instagrapi settings to %s", session_file)
        cl.dump_settings(session_file)

        # 3) users_data update (same tumhara code)
        logging.info("[PLOGIN] Updating users_data for user_id=%s", user_id)
        if user_id not in users_data:
            users_data[user_id] = {
                'accounts': [],
                'default': None,
                'pairs': None,
                'switch_minutes': 10,
                'threads': 1,
            }
            save_user_data(user_id, users_data[user_id])

        data = users_data[user_id]
        found = False
        for i, acc in enumerate(data['accounts']):
            if acc.get('ig_username', '').strip().lower() == username:
                acc['password'] = password
                acc['storage_state'] = state
                data['default'] = i
                found = True
                logging.info("[PLOGIN] Updated existing account index=%s", i)
                break

        if not found:
            data['accounts'].append({
                'ig_username': username,
                'password': password,
                'storage_state': state,
            })
            data['default'] = len(data['accounts']) - 1
            logging.info("[PLOGIN] Added new account, total=%s", len(data['accounts']))

        save_user_data(user_id, data)

        await update.message.reply_text("✅ Playwright login successful! Sessions saved. 🎉")

    except ValueError as ve:
        err_msg = str(ve)
        logging.error("[PLOGIN] ValueError: %s", err_msg)

        # Specific errors like ERROR_010, ERROR_OTP, timeouts etc
        await update.message.reply_text(f"❌ Login failed: {err_msg}")

    except Exception as e:
        logging.exception("[PLOGIN] Unexpected exception in plogin_get_password")
        await update.message.reply_text(f"❌ Unexpected error: {e}")

    return ConversationHandler.END

# --- /slogin handlers ---
async def slogin_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return ConversationHandler.END
    await update.message.reply_text("🔑 Enter session ID: ")
    return SLOG_SESSION

async def slogin_get_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    sessionid = update.message.text.strip()
    user_id = update.effective_user.id
    temp_file = f"temp_session_{user_id}_{int(time.time())}.json"
    
    try:
        # Store the session ID directly without validation
        # Validation will happen when actually using the session
        with open(temp_file, 'w') as f:
            json.dump({"sessionid": sessionid}, f)
        
        context.user_data['temp_session_file'] = temp_file
        await update.message.reply_text("✅ Session ID received! 📝 Enter username to save:")
        return SLOG_USERNAME
    except Exception as e:
        os.remove(temp_file) if os.path.exists(temp_file) else None
        await update.message.reply_text(f"❌ Error: {str(e)}")
        return ConversationHandler.END

async def slogin_get_username(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        username = update.message.text.strip().lower()
        user_id = update.effective_user.id
        temp_file = context.user_data['temp_session_file']
        session_file = f"sessions/{user_id}_{username}_session.json"
        os.makedirs('sessions', exist_ok=True)
        
        # Load the sessionid from temp file
        with open(temp_file, 'r') as f:
            session_data = json.load(f)
        
        sessionid = session_data.get('sessionid', '')
        
        # Try to initialize instagrapi client with the sessionid
        cl = Client()
        try:
            cl.login_by_sessionid(sessionid)
            cl.dump_settings(session_file)
        except Exception as login_err:
            os.remove(temp_file) if os.path.exists(temp_file) else None
            await update.message.reply_text(f"❌ Failed to validate session with Instagram: {str(login_err)}\n\nThe session ID may be invalid or expired.")
            return ConversationHandler.END

        # Generate playwright state from instagrapi settings
        try:
            settings = json.load(open(session_file))
            state = get_storage_state_from_instagrapi(settings)
        except Exception:
            # Fallback: create minimal state if conversion fails
            state = {
                "cookies": [],
                "origins": [{"origin": "https://www.instagram.com", "localStorage": []}]
            }
        
        playwright_file = f"sessions/{user_id}_{username}_state.json"
        with open(playwright_file, 'w') as f:
            json.dump(state, f)

        # Save to user data
        if user_id not in users_data:
            users_data[user_id] = {'accounts': [], 'default': None, 'pairs': None, 'switch_minutes': 10, 'threads': 1}
            save_user_data(user_id, users_data[user_id])
        data = users_data[user_id]
        found = False
        for i, acc in enumerate(data['accounts']):
            if acc.get('ig_username', '').strip().lower() == username:
                acc['password'] = ''  # No password for session
                acc['storage_state'] = state
                data['default'] = i
                found = True
                break
        if not found:
            data['accounts'].append({'ig_username': username, 'password': '', 'storage_state': state})
            data['default'] = len(data['accounts']) - 1
        save_user_data(user_id, data)

        os.remove(temp_file) if os.path.exists(temp_file) else None
        await update.message.reply_text(f"✅ Session saved for {username}! Playwright & Instagrapi files created. 🎉")
        return ConversationHandler.END
    except Exception as e:
        temp_file = context.user_data.get('temp_session_file', '')
        if os.path.exists(temp_file):
            os.remove(temp_file)
        await update.message.reply_text(f"❌ Error saving session: {str(e)}")
        return ConversationHandler.END

async def viewmyac(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if user_id not in users_data:
        await update.message.reply_text("❌ You haven't saved any account. Use /login to save one. ❌")
        return
    data = users_data[user_id]
    msg = "👀 Your saved account list 👀\n"
    for i, acc in enumerate(data['accounts']):
        default = " (default) ⭐" if data['default'] == i else ""
        msg += f"{i+1}. {acc['ig_username']}{default}\n"
    await update.message.reply_text(msg)

async def setig(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❗ Usage: /setig <number> ❗")
        return
    num = int(context.args[0]) - 1
    if user_id not in users_data:
        await update.message.reply_text("❌ No accounts saved. ❌")
        return
    data = users_data[user_id]
    if num < 0 or num >= len(data['accounts']):
        await update.message.reply_text("⚠️ Invalid number. ⚠️")
        return
    data['default'] = num
    save_user_data(user_id, data)
    acc = data['accounts'][num]['ig_username']
    await update.message.reply_text(f"✅ {num+1}. {acc} now is your default account. ⭐")

async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if not context.args:
        await update.message.reply_text("❗ Usage: /logout <username> ❗")
        return
    username = context.args[0].strip()
    if user_id not in users_data:
        await update.message.reply_text("❌ No accounts saved. ❌")
        return
    data = users_data[user_id]
    for i, acc in enumerate(data['accounts']):
        if acc['ig_username'] == username:
            del data['accounts'][i]
            if data['default'] == i:
                data['default'] = 0 if data['accounts'] else None
            elif data['default'] > i:
                data['default'] -= 1
            # Update pairs if exists
            if data['pairs']:
                pl = data['pairs']['list']
                if username in pl:
                    pl.remove(username)
                    if not pl:
                        data['pairs'] = None
                    else:
                        data['pairs']['default_index'] = 0
            break
    else:
        await update.message.reply_text("⚠️ Account not found. ⚠️")
        return
    save_user_data(user_id, data)
    session_file = f"sessions/{user_id}_{username}_session.json"
    state_file = f"sessions/{user_id}_{username}_state.json"
    if os.path.exists(session_file):
        os.remove(session_file)
    if os.path.exists(state_file):
        os.remove(state_file)
    await update.message.reply_text(f"✅ Logged out and removed {username}. Files deleted. ✅")

# New commands
async def pair_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther⚠️")
        return
    if not context.args:
        await update.message.reply_text("❗ Usage: /pair iguser1-iguser2-iguser3 ❗")
        return
    arg_str = '-'.join(context.args)
    us = [u.strip() for u in arg_str.split('-') if u.strip()]
    if len(us) < 2:
        await update.message.reply_text("❗ Provide at least one more account. ❗")
        return
    if user_id not in users_data or not users_data[user_id]['accounts']:
        await update.message.reply_text("❌ No accounts saved. Use /login first. ❌")
        return
    data = users_data[user_id]
    accounts_set = {acc['ig_username'] for acc in data['accounts']}
    missing = [u for u in us if u not in accounts_set]
    if missing:
        await update.message.reply_text(f"⚠️ Can't find that account: {missing[0]}. Save it again with /login. ⚠️")
        return
    data['pairs'] = {'list': us, 'default_index': 0}
    # Set default to first in pair
    first_u = us[0]
    for i, acc in enumerate(data['accounts']):
        if acc['ig_username'] == first_u:
            data['default'] = i
            break
    save_user_data(user_id, data)
    await update.message.reply_text(f"✅ Pair created! {len(us)} accounts saved.\nDefault: {first_u} ⭐\nUse /attack to start sending messages with pairing & switching.")

async def unpair_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return

    if user_id not in users_data or not users_data[user_id].get('pairs'):
        await update.message.reply_text("❌ No active pair found. Use /pair first. ❌")
        return

    data = users_data[user_id]
    pair_info = data['pairs']
    pair_list = pair_info['list']

    # --- no arguments case ---
    if not context.args:
        msg = "👥 Current pair list:\n"
        for i, u in enumerate(pair_list, 1):
            mark = "⭐" if i - 1 == pair_info.get('default_index', 0) else ""
            msg += f"{i}. {u} {mark}\n"
        msg += "\nUse `/unpair all` to remove all pairs or `/unpair <username>` to remove one."
        await update.message.reply_text(msg)
        return

    arg = context.args[0].strip().lower()

    # --- unpair all ---
    if arg == "all":
        data['pairs'] = None
        save_user_data(user_id, data)
        await update.message.reply_text("🧹 All paired accounts removed successfully.")
        return

    # --- unpair specific account ---
    target = arg
    if target not in pair_list:
        await update.message.reply_text(f"⚠️ {target} is not in current pair list. ⚠️")
        return

    pair_list.remove(target)
    if not pair_list:
        data['pairs'] = None
        msg = f"✅ Removed {target}. No pairs left."
    else:
        # adjust default index if needed
        if pair_info.get('default_index', 0) >= len(pair_list):
            pair_info['default_index'] = 0
        msg = f"✅ Removed {target}. Remaining pairs: {', '.join(pair_list)}"

    save_user_data(user_id, data)
    await update.message.reply_text(msg)

async def switch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❗ Usage: /switch <minutes> ❗")
        return
    min_ = int(context.args[0])
    data = users_data[user_id]
    if not data.get('pairs') or len(data['pairs']['list']) < 2:
        await update.message.reply_text("⚠️ No pair found. Use /pair first. ⚠️")
        return
    if min_ < 5:
        await update.message.reply_text("⚠️ Minimum switch interval is 5 minutes. ⚠️")
        return
    data['switch_minutes'] = min_
    save_user_data(user_id, data)
    await update.message.reply_text(f"⏱️ Switch interval set to {min_} minutes.")

async def threads_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❗ Usage: /threads <1-5> ❗")
        return
    n = int(context.args[0])
    if n < 1 or n > 5:
        await update.message.reply_text("⚠️ threads must be between 1 and 5. ⚠️")
        return
    if user_id not in users_data:
        users_data[user_id] = {'accounts': [], 'default': None, 'pairs': None, 'switch_minutes': 10, 'threads': 1}
        save_user_data(user_id, users_data[user_id])
    data = users_data[user_id]
    data['threads'] = n
    save_user_data(user_id, data)
    await update.message.reply_text(f"🔁 Threads set to {n}.")

async def viewpref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if user_id not in users_data:
        await update.message.reply_text("❌ No data. Use /login. ❌")
        return
    data = users_data[user_id]
    saved_accounts = ', '.join([acc['ig_username'] for acc in data['accounts']])
    msg = "🔧 Your bot preferences:\n"
    if data.get('pairs'):
        pl = data['pairs']['list']
        msg += f"Pairs: yes — {len(pl)} accounts\n"
        default_idx = data['pairs']['default_index']
        default_u = pl[default_idx]
        msg += f"Default: {default_u} ⭐\n"
    else:
        msg += "Pairs: no\n"
    switch_min = data.get('switch_minutes', 10)
    msg += f"⏱️ Switch interval: {switch_min} minutes\n"
    threads = data.get('threads', 1)
    msg += f"🧵 Threads: {threads}\n"
    msg += f"👤 Saved accounts: {saved_accounts}\n"
    # Check running attacks
    tasks = users_tasks.get(user_id, [])
    running_attacks = [t for t in tasks if t.get('type') == 'message_attack' and t['status'] == 'running' and t['proc'].poll() is None]
    if running_attacks:
        task = running_attacks[0]  # Assume one
        pid = task['pid']
        ttype = task['target_type']
        tdisplay = task['target_display']
        disp = f"dm -> @{tdisplay}" if ttype == 'dm' else tdisplay
        msg += f"\nActive attack PID {pid} ({disp})\n"
        msg += "Spamming...!\n"
        pair_list = task['pair_list']
        curr_idx = task['pair_index']
        curr_u = pair_list[curr_idx]
        for u in pair_list:
            if u == curr_u:
                msg += f"using - {u}\n"
            else:
                msg += f"cooldown - {u}\n"
    await update.message.reply_text(msg)

MODE, SELECT_GC, TARGET, MESSAGES = range(4)

async def attack_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return ConversationHandler.END
    if user_id not in users_data or not users_data[user_id]['accounts']:
        await update.message.reply_text("❗ Please /login first. ❗")
        return ConversationHandler.END
    data = users_data[user_id]
    if data['default'] is None:
        data['default'] = 0
        save_user_data(user_id, data)
    await update.message.reply_text("Where you want to send msgs ? dm or gc")
    return MODE

async def get_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.message.text.lower().strip()
    data = users_data[user_id]

    if 'dm' in text:
        context.user_data['mode'] = 'dm'
        await update.message.reply_text("Enter target username (for DM):")
        return TARGET

    elif 'gc' in text:
        acc = data['accounts'][data['default']]
        fetch_msg = await update.message.reply_text("🔍 Fetching last 10 GC threads...")

        user_fetching.add(user_id)
        try:
            groups, new_state = await asyncio.to_thread(
                list_group_chats,
                user_id,
                acc['storage_state'],
                acc['ig_username'],
                acc['password'],
                max_groups=10,
                amount=10
            )
        finally:
            user_fetching.discard(user_id)
            try:
                await fetch_msg.delete()
            except:
                pass

        # ✅ Agar beech me /cancel aaya tha to result ignore karo
        if user_id in user_cancel_fetch:
            user_cancel_fetch.discard(user_id)
            await update.message.reply_text("❌ Fetching cancelled. Ignoring result. ❌")
            return ConversationHandler.END

        if new_state != acc['storage_state']:
            acc['storage_state'] = new_state
            save_user_data(user_id, data)

        context.user_data['groups'] = groups
        context.user_data['mode'] = 'gc'

        if not groups:
            await update.message.reply_text("❌ No group chats found. ❌")
            return ConversationHandler.END

        msg = "🔢 Select a GC by number: 🔢\n"
        for i, g in enumerate(groups):
            msg += f"{i+1}. {g['display']}\n"

        await update.message.reply_text(msg)
        return SELECT_GC

    else:
        await update.message.reply_text("Please reply with 'dm' or 'gc'")
        return MODE

async def select_gc_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.message.text.strip()
    try:
        num = int(text) - 1
        groups = context.user_data['groups']
        if 0 <= num < len(groups):
            g = groups[num]
            context.user_data['thread_url'] = g['url']
            context.user_data['target_display'] = g['display']
            await update.message.reply_text("Send messages like: msg1 & msg2 & msg3 or upload .txt file")
            return MESSAGES
        else:
            await update.message.reply_text("⚠️ Invalid number. Please select 1-{}. ⚠️".format(len(groups)))
            return SELECT_GC
    except ValueError:
        await update.message.reply_text("⚠️ Please send a number. ⚠️")
        return SELECT_GC

async def get_target_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    target_u = update.message.text.strip().lstrip('@')
    if not target_u:
        await update.message.reply_text("⚠️ Invalid username. ⚠️")
        return TARGET
    context.user_data['target_display'] = target_u
    data = users_data[user_id]
    acc = data['accounts'][data['default']]
    thread_url = await asyncio.to_thread(
        get_dm_thread_url, user_id, acc['ig_username'], acc['password'], target_u
    )
    if not thread_url:
        await update.message.reply_text("❌ Could not lock thread id with default account.")
        return ConversationHandler.END
    context.user_data['thread_url'] = thread_url
    await update.message.reply_text("Send messages like: msg1 & msg2 & msg3 or upload .txt file")
    return MESSAGES
    
async def get_messages_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    document = update.message.document

    if not document:
        await update.message.reply_text("❌ Please upload a .txt file.")
        return ConversationHandler.END

    file = await document.get_file()

    import uuid, os
    randomid = str(uuid.uuid4())[:8]
    names_file = f"{user_id}_{randomid}.txt"

    # Save uploaded .txt file
    await file.download_to_drive(names_file)

    # store file path in context so get_messages can use it
    context.user_data['uploaded_names_file'] = names_file

    # Reuse same logic as text handler
    return await get_messages(update, context)

async def get_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id

    import uuid, os, json, time, random, sys

    # Check if we came from file upload handler
    uploaded_file = context.user_data.pop('uploaded_names_file', None)

    if uploaded_file and os.path.exists(uploaded_file):
        # Use already saved .txt file from upload
        names_file = uploaded_file
        raw_text = f"[USING_UPLOADED_FILE:{os.path.basename(uploaded_file)}]"
        logging.debug("USING UPLOADED FILE: %r", uploaded_file)
    else:
        # Normal text input flow
        raw_text = (update.message.text or "").strip()
        logging.debug("RAW MESSAGES INPUT: %r", raw_text)

        # Normalize to handle fullwidth & etc.
        text = unicodedata.normalize("NFKC", raw_text)

        # Always make a temp file
        randomid = str(uuid.uuid4())[:8]
        names_file = f"{user_id}_{randomid}.txt"

        # ✅ Write raw text directly so msgb.py handles splitting correctly
        try:
            with open(names_file, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            await update.message.reply_text(f"❌ Error creating file: {e}")
            return ConversationHandler.END

    # --- Below part unchanged (keeps rotation, task limits, etc.) ---

    # --- Below part unchanged (keeps rotation, task limits, etc.) ---
    data = users_data[user_id]
    pairs = data.get('pairs')
    pair_list = pairs['list'] if pairs else [data['accounts'][data['default']]['ig_username']]
    if len(pair_list) == 1:
        warning = "⚠️ Warning: You may get chat ban if you use a single account too long. Use /pair to make multi-account rotation.\n\n"
    else:
        warning = ""
    switch_minutes = data.get('switch_minutes', 10)
    threads_n = data.get('threads', 1)
    tasks = users_tasks.get(user_id, [])
    running_msg = [t for t in tasks if t.get('type') == 'message_attack' and t['status'] == 'running' and t['proc'].poll() is None]
    if len(running_msg) >= 5:
        await update.message.reply_text("⚠️ Max 5 message attacks running. Stop one first. ⚠️")
        if os.path.exists(names_file):
            os.remove(names_file)
        return ConversationHandler.END

    thread_url = context.user_data['thread_url']
    target_display = context.user_data['target_display']
    target_mode = context.user_data['mode']
    start_idx = pairs['default_index'] if pairs else 0
    start_u = pair_list[start_idx]
    start_acc = next(acc for acc in data['accounts'] if acc['ig_username'] == start_u)
    start_pass = start_acc['password']
    start_u = start_u.strip().lower()
    state_file = f"sessions/{user_id}_{start_u}_state.json"
    if not os.path.exists(state_file):
        with open(state_file, 'w') as f:
            json.dump(start_acc['storage_state'], f)

    python_exec = sys.executable
    # Locate sender script (supports both 'swiper.py' and old 'sesion msg.py')
    script_path = _find_sender_script()
    cmd = [
        python_exec, script_path,
        "--username", start_u,
        "--password", start_pass,
        "--thread-url", thread_url,
        "--names", names_file,
        "--tabs", str(threads_n),
        "--headless", "true",
        "--storage-state", state_file
    ]
    task_id = str(uuid.uuid4())
    log_file = f"attack_{task_id}.log"
    logging.info("Starting attack subprocess", extra={"cmd": cmd, "log_file": log_file})
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=f)
    except FileNotFoundError as e:
        logging.exception("Failed to start attack subprocess")
        await update.message.reply_text(f"❌ Failed to start attack process: {e}")
        if os.path.exists(names_file):
            os.remove(names_file)
        return ConversationHandler.END
    except Exception as e:
        logging.exception("Unexpected error starting attack subprocess")
        await update.message.reply_text(f"❌ Error starting attack: {e}")
        if os.path.exists(names_file):
            os.remove(names_file)
        return ConversationHandler.END
    running_processes[proc.pid] = proc
    pid = proc.pid
    task = {
        "id": task_id,
        "user_id": user_id,
        "type": "message_attack",
        "pair_list": pair_list,
        "pair_index": start_idx,
        "switch_minutes": switch_minutes,
        "threads": threads_n,
        "names_file": names_file,
        "target_thread_url": thread_url,
        "target_type": target_mode,
        "target_display": target_display,
        "last_switch_time": time.time(),
        "status": "running",
        "cmd": cmd,
        "pid": pid,
        "display_pid": pid,
        "proc_list": [pid],
        "proc": proc,
        "start_time": time.time()
    }
    persistent_tasks.append(task)
    save_persistent_tasks()
    tasks.append(task)
    users_tasks[user_id] = tasks
    logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Message attack start user={user_id} task={task_id} target={target_display} pid={pid}")

    status = "Spamming...!\n"
    curr_u = pair_list[task['pair_index']]
    for u in pair_list:
        if u == curr_u:
            status += f"using - {u}\n"
        else:
            status += f"cooldown - {u}\n"
    status += f"To stop 🛑 type /stop {task['display_pid']} or /stop all to stop all processes."

    sent_msg = await update.message.reply_text(warning + status)
    task['status_chat_id'] = update.message.chat_id
    task['status_msg_id'] = sent_msg.message_id
    return ConversationHandler.END

def load_persistent_tasks():
    global persistent_tasks
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r') as f:
            persistent_tasks = json.load(f)
    else:
        persistent_tasks = []

def save_persistent_tasks():
    """
    Safely write persistent_tasks to TASKS_FILE.
    Removes runtime-only values (like 'proc') and ensures JSON-safe data.
    """
    safe_list = []
    for t in persistent_tasks:
        cleaned = {}
        for k, v in t.items():
            if k == 'proc':
                continue
            if isinstance(v, (int, float, str, bool, dict, list, type(None))):
                cleaned[k] = v
            else:
                try:
                    json.dumps(v)
                    cleaned[k] = v
                except Exception:
                    cleaned[k] = str(v)
        safe_list.append(cleaned)

    temp_file = TASKS_FILE + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(safe_list, f, indent=2)
    os.replace(temp_file, TASKS_FILE)

def mark_task_stopped_persistent(task_id: str):
    global persistent_tasks
    for task in persistent_tasks:
        if task['id'] == task_id:
            task['status'] = 'stopped'
            save_persistent_tasks()
            break

def update_task_pid_persistent(task_id: str, new_pid: int):
    global persistent_tasks
    for task in persistent_tasks:
        if task['id'] == task_id:
            task['pid'] = new_pid
            save_persistent_tasks()
            break

def mark_task_completed_persistent(task_id: str):
    global persistent_tasks
    for task in persistent_tasks:
        if task['id'] == task_id:
            task['status'] = 'completed'
            save_persistent_tasks()
            break

def restore_tasks_on_start():
    load_persistent_tasks()
    print(f"🔄 Restoring {len([t for t in persistent_tasks if t.get('type') == 'message_attack' and t['status'] == 'running'])} running message attacks...")
    for task in persistent_tasks[:]:
        if task.get('type') == 'message_attack' and task['status'] == 'running':
            old_pid = task['pid']
            try:
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(1)
            except OSError:
                pass  # Already dead
            user_id = task['user_id']
            data = users_data.get(user_id)
            if not data:
                mark_task_stopped_persistent(task['id'])
                continue
            pair_list = task['pair_list']
            curr_idx = task['pair_index']
            curr_u = pair_list[curr_idx]
            curr_acc = None
            for acc in data['accounts']:
                if acc['ig_username'] == curr_u:
                    curr_acc = acc
                    break
            if not curr_acc:
                mark_task_stopped_persistent(task['id'])
                continue
            curr_pass = curr_acc['password']
            curr_u = curr_u.strip().lower()
            state_file = f"sessions/{user_id}_{curr_u}_state.json"
            if not os.path.exists(state_file):
                with open(state_file, 'w') as f:
                    json.dump(curr_acc['storage_state'], f)
            names_file = task['names_file']
            if not os.path.exists(names_file):
                # Recreate if missing? But skip for now
                mark_task_stopped_persistent(task['id'])
                continue
            python_exec = sys.executable
            script_path = _find_sender_script()
            cmd = [
                python_exec, script_path,
                "--username", curr_u,
                "--password", curr_pass,
                "--thread-url", task['target_thread_url'],
                "--names", names_file,
                "--tabs", str(task['threads']),
                "--headless", "true",
                "--storage-state", state_file
            ]
            logging.info("Restoring attack subprocess", extra={"cmd": cmd})
            try:
                proc = subprocess.Popen(cmd)
                # Register runtime map
                running_processes[proc.pid] = proc
                new_pid = proc.pid
                update_task_pid_persistent(task['id'], new_pid)
                mem_task = task.copy()
                mem_task['proc'] = proc
                mem_task['proc_list'] = [proc.pid]
                mem_task['display_pid'] = task.get('display_pid', proc.pid)
                if user_id not in users_tasks:
                    users_tasks[user_id] = []
                users_tasks[user_id].append(mem_task)
                print(f"✅ Restored message attack {task['id']} for {task['target_display']} | New PID: {new_pid}")
            except Exception as e:
                logging.exception(f"❌ Failed to restore message attack {task['id']}")
                mark_task_stopped_persistent(task['id'])
    save_persistent_tasks()
    print("✅ Task restoration complete!")

async def send_resume_notification(user_id: int, task: Dict):
    ttype = task['target_type']
    tdisplay = task['target_display']
    disp = f"dm -> @{tdisplay}" if ttype == 'dm' else tdisplay
    msg = f"🔄 Attack auto resumed! New PID: {task['pid']} ({disp})\n"
    pair_list = task['pair_list']
    curr_idx = task['pair_index']
    curr_u = pair_list[curr_idx]
    for u in pair_list:
        if u == curr_u:
            msg += f"using - {u}\n"
        else:
            msg += f"cooldown - {u}\n"
    try:
        await APP.bot.send_message(chat_id=user_id, text=msg)
    except Exception as e:
        logging.warning(f"Failed to send resume notification to user {user_id}: {e}")
        # Continue without crashing - user might have blocked the bot

def get_switch_update(task: Dict) -> str:
    pair_list = task['pair_list']
    curr_idx = task['pair_index']
    curr_u = pair_list[curr_idx]
    lines = []
    for u in pair_list:
        if u == curr_u:
            lines.append(f"using - {u}")
        else:
            lines.append(f"cooldown - {u}")
    return '\n'.join(lines)

def switch_task_sync(task: Dict):
    user_id = task['user_id']

    # Keep reference to old proc (don't terminate it yet)
    try:
        old_proc = task.get('proc')
        old_pid = task.get('pid')
    except Exception:
        old_proc = None
        old_pid = task.get('pid')

    # Advance index first so new account is chosen
    task['pair_index'] = (task['pair_index'] + 1) % len(task['pair_list'])
    next_u = task['pair_list'][task['pair_index']]
    data = users_data.get(user_id)
    if not data:
        logging.error(f"No users_data for user {user_id} during switch")
        return

    next_acc = next((a for a in data['accounts'] if a['ig_username'] == next_u), None)
    if not next_acc:
        logging.error(f"Can't find account {next_u} for switch")
        try:
            asyncio.run_coroutine_threadsafe(
                APP.bot.send_message(user_id, f"can't find thread Id - {next_u}"),
                LOOP
            )
        except Exception:
            pass
        return

    next_pass = next_acc['password']
    next_state_file = f"sessions/{user_id}_{next_u}_state.json"
    if not os.path.exists(next_state_file):
        try:
            with open(next_state_file, 'w') as f:
                json.dump(next_acc.get('storage_state', {}), f)
        except Exception as e:
            logging.error(f"Failed to write state file for {next_u}: {e}")

    # Launch new process FIRST so overlap prevents downtime
    python_exec = sys.executable
    script_path = _find_sender_script()
    new_cmd = [
        python_exec, script_path,
        "--username", next_u,
        "--password", next_pass,
        "--thread-url", task['target_thread_url'],
        "--names", task['names_file'],
        "--tabs", str(task['threads']),
        "--headless", "true",
        "--storage-state", next_state_file
    ]
    try:
        new_proc = subprocess.Popen(new_cmd)
    except Exception as e:
        logging.exception(f"Failed to launch new proc for switch to {next_u}")
        return

    # Append new to proc_list
    task['proc_list'].append(new_proc.pid)

    # Register new proc and update task/persistent info
    running_processes[new_proc.pid] = new_proc
    task['cmd'] = new_cmd
    task['pid'] = new_proc.pid
    task['proc'] = new_proc
    task['last_switch_time'] = time.time()
    try:
        update_task_pid_persistent(task['id'], task['pid'])
    except Exception as e:
        logging.error(f"Failed to update persistent pid for task {task.get('id')}: {e}")

    # Give old proc a short cooldown window before killing it (avoid downtime)
    if old_proc and old_pid != new_proc.pid:
        try:
            # Allow overlap for a short cooldown
            time.sleep(5)
            try:
                old_proc.terminate()
            except Exception:
                pass
            # wait a bit for graceful shutdown
            time.sleep(2)
            if old_proc.poll() is None:
                try:
                    old_proc.kill()
                except Exception:
                    pass
            # Remove old from proc_list and running_processes
            if old_pid in task['proc_list']:
                task['proc_list'].remove(old_pid)
            if old_pid in running_processes:
                running_processes.pop(old_pid, None)
        except Exception as e:
            logging.error(f"Error while stopping old proc after switch: {e}")

    # Send/update status message (edit if message id present)
    try:
        chat_id = task.get('status_chat_id', user_id)
        msg_id = task.get('status_msg_id')
        text = "Spamming...!\n" + get_switch_update(task)
        text += f"\nTo stop 🛑 type /stop {task['display_pid']} or /stop all to stop all processes."
        if msg_id:
            asyncio.run_coroutine_threadsafe(
                APP.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text),
                LOOP
            )
        else:
            asyncio.run_coroutine_threadsafe(
                APP.bot.send_message(chat_id=chat_id, text=text),
                LOOP
            )
    except Exception as e:
        logging.error(f"Failed to update status message: {e}")

def switch_monitor():
    while True:
        time.sleep(30)
        for user_id in list(users_tasks):
            if user_id not in users_tasks:
                continue
            for task in users_tasks[user_id]:
                if task.get('type') == 'message_attack' and task['status'] == 'running' and task['proc'].poll() is None:
                    due_time = task['last_switch_time'] + task['switch_minutes'] * 60
                    if time.time() >= due_time:
                        if len(task['pair_list']) > 1:
                            switch_task_sync(task)

async def engage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Command: /engage <thread_url>
    PURE ENGAGEMENT MODE: Only reacts and swipes replies to messages.
    No message sending - 100% focus on reactions!
    """
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    
    if not context.args:
        await update.message.reply_text("❗ Usage: /engage <thread_url>\nExample: /engage https://www.instagram.com/direct/t/123456/")
        return
    
    thread_url = context.args[0].strip()
    if 'instagram.com' not in thread_url:
        await update.message.reply_text("⚠️ Invalid Instagram thread URL ⚠️")
        return
    
    if user_id not in users_data or not users_data[user_id]['accounts']:
        await update.message.reply_text("❗ Please /login first. ❗")
        return ConversationHandler.END
    
    data = users_data[user_id]
    if data['default'] is None:
        data['default'] = 0
        save_user_data(user_id, data)
    
    acc = data['accounts'][data['default']]
    
    # Instead of launching immediately, ask user for the reply text to use.
    # Save necessary info in user_data and start a conversation to collect the text.
    try:
        # Ensure we pass a file path for storage_state (some accounts store dicts)
        storage_state_val = acc.get('storage_state')
        storage_path = None
        if isinstance(storage_state_val, dict):
            # write to temporary file in sessions/
            tmp_name = f"sessions/{user_id}_{acc['ig_username']}_state_tmp_{int(time.time())}.json"
            try:
                with open(tmp_name, 'w') as tf:
                    json.dump(storage_state_val, tf)
                storage_path = tmp_name
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to write storage state: {e}")
                return ConversationHandler.END
        else:
            storage_path = storage_state_val

        if not storage_path or not isinstance(storage_path, str):
            await update.message.reply_text("❌ Invalid storage state for account. Please re-login with /plogin or /slogin.")
            return ConversationHandler.END

        # store for follow-up handler
        context.user_data['engage_storage_path'] = storage_path
        context.user_data['engage_thread_url'] = thread_url
        context.user_data['engage_account'] = acc['ig_username']

        await update.message.reply_text("📣 What text should I send as the reply? Send the text message now.")
        return ENGAGE_TEXT
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
        return ConversationHandler.END

async def engage_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Follow-up handler for /engage conversation. Receives the reply text from the user
    and spawns engage.py with --reply-text pointing to the provided text.
    """
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return ConversationHandler.END

    reply_text = (update.message.text or "").strip()
    if not reply_text:
        await update.message.reply_text("❗ Reply text cannot be empty. Please send the text you want me to reply with.")
        return ENGAGE_TEXT

    storage_path = context.user_data.get('engage_storage_path')
    thread_url = context.user_data.get('engage_thread_url')
    account = context.user_data.get('engage_account')

    if not storage_path or not thread_url:
        await update.message.reply_text("❌ Missing session info. Please re-run /engage <thread_url>.")
        return ConversationHandler.END

    try:
        # Build command to spawn engage.py
        cmd = [
            "python3", "engage.py",
            "--thread-url", thread_url,
            "--storage-state", storage_path,
            "--reply-text", reply_text,
            "--headless", "true"
        ]
        proc = subprocess.Popen(cmd)
        running_processes[proc.pid] = proc
        pid = proc.pid
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "user_id": user_id,
            "type": "engagement",
            "target_thread_url": thread_url,
            "target_display": account,
            "status": "running",
            "cmd": cmd,
            "pid": pid,
            "display_pid": pid,
            "proc": proc,
            "start_time": time.time()
        }
        persistent_tasks.append(task)
        save_persistent_tasks()
        if user_id not in users_tasks:
            users_tasks[user_id] = []
        users_tasks[user_id].append(task)

        await update.message.reply_text(f"✅ Engagement started. PID: {pid} | Reply text set.")
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Engagement start user={user_id} task={task_id} pid={pid} reply_text={reply_text}")
        return ConversationHandler.END
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to start engagement: {e}")
        logging.error(f"Failed to spawn engage.py: {e}")
        return ConversationHandler.END

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if not context.args:
        await update.message.reply_text("❗ Usage: /stop <PID> or /stop all ❗")
        return
    arg = context.args[0]
    if user_id not in users_tasks or not users_tasks[user_id]:
        await update.message.reply_text("❌ No tasks running. ❌")
        return
    tasks = users_tasks[user_id]
    if arg == 'all':
        stopped_count = 0
        for task in tasks[:]:
            proc = task['proc']
            proc.terminate()
            await asyncio.sleep(3)
            if proc.poll() is None:
                proc.kill()
            # Remove from runtime map if present
            pid = task.get('pid')
            if pid in running_processes:
                running_processes.pop(pid, None)
            if task.get('type') == 'message_attack' and 'names_file' in task:
                names_file = task['names_file']
                if os.path.exists(names_file):
                    os.remove(names_file)
            logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Task stop user={user_id} task={task['id']}")
            mark_task_stopped_persistent(task['id'])
            tasks.remove(task)
            stopped_count += 1
        await update.message.reply_text(f"🛑 Stopped all your tasks! ({stopped_count}) 🛑")
    elif arg.isdigit():
        pid_to_stop = int(arg)
        stopped_task = None

        # 1) Try users_tasks by display_pid
        for task in tasks[:]:
            if task.get('display_pid') == pid_to_stop:
                proc_list = task.get('proc_list', [])
                for backend_pid in proc_list:
                    backend_proc = running_processes.get(backend_pid)
                    if backend_proc:
                        try:
                            backend_proc.terminate()
                        except Exception:
                            pass
                        await asyncio.sleep(3)
                        if backend_proc.poll() is None:
                            try:
                                backend_proc.kill()
                            except Exception:
                                pass
                    else:
                        try:
                            os.kill(backend_pid, signal.SIGTERM)
                        except Exception:
                            pass
                for backend_pid in proc_list:
                    running_processes.pop(backend_pid, None)
                mark_task_stopped_persistent(task['id'])
                if 'names_file' in task and os.path.exists(task['names_file']):
                    os.remove(task['names_file'])
                stopped_task = task
                tasks.remove(task)
                await update.message.reply_text(f"🛑 Stopped task {pid_to_stop}!")
                break

        # 2) If not found, fallback to runtime map for single pid
        if not stopped_task:
            proc = running_processes.get(pid_to_stop)
            if proc:
                try: proc.terminate()
                except Exception: pass
                await asyncio.sleep(2)
                if proc.poll() is None:
                    try: proc.kill()
                    except Exception: pass
                running_processes.pop(pid_to_stop, None)
                for t in persistent_tasks:
                    if t.get('pid') == pid_to_stop:
                        mark_task_stopped_persistent(t['id'])
                        break
                await update.message.reply_text(f"🛑 Stopped task {pid_to_stop}!")
                return

        if not stopped_task:
            await update.message.reply_text("⚠️ Task not found. ⚠️")
    else:
        await update.message.reply_text("❗ Usage: /stop <PID> or /stop all ❗")
    users_tasks[user_id] = tasks

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    if user_id not in users_tasks or not users_tasks[user_id]:
        await update.message.reply_text("❌ No ongoing tasks. ❌")
        return
    tasks = users_tasks[user_id]
    active_tasks = []
    for t in tasks:
        if t['proc'].poll() is None:
            active_tasks.append(t)
        else:
            mark_task_completed_persistent(t['id'])
    users_tasks[user_id] = active_tasks
    if not active_tasks:
        await update.message.reply_text("❌ No active tasks. ❌")
        return
    msg = "📋 Ongoing tasks 📋\n"
    for task in active_tasks:
        tdisplay = task.get('target_display', 'Unknown')
        ttype = task.get('type', 'unknown')
        preview = tdisplay[:20] + '...' if len(tdisplay) > 20 else tdisplay
        display_pid = task.get('display_pid', task['pid'])
        msg += f"PID {display_pid} — {preview} ({ttype})\n"
    await update.message.reply_text(msg)

async def usg_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⚠️ You are not authorised to use, dm owner to gain access! @spyther ⚠️")
        return
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_total = mem.total / (1024 ** 3)
    msg = f"🖥️ CPU Usage: {cpu:.1f}%\n💾 RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({mem.percent:.1f}%)"
    await update.message.reply_text(msg)

async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_fetching:
        user_fetching.discard(user_id)
        await update.message.reply_text("❌ Fetching cancelled! 🚫")
    else:
        await update.message.reply_text("No active fetching to cancel. 😊")

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⚠️ you are not an admin ⚠️")
        return
    if len(context.args) != 1:
        await update.message.reply_text("❗ Usage: /add <tg_id> ❗")
        return
    try:
        tg_id = int(context.args[0])
        if any(u['id'] == tg_id for u in authorized_users):
            await update.message.reply_text("❗ User already added. ❗")
            return
        authorized_users.append({'id': tg_id, 'username': ''})
        save_authorized()
        await update.message.reply_text(f"➕ Added {tg_id} as authorized user. ➕")
    except:
        await update.message.reply_text("⚠️ Invalid tg_id. ⚠️")

async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⚠️ you are not an admin ⚠️")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❗ Usage: /remove <tg_id> ❗")
        return
    tg_id = int(context.args[0])
    global authorized_users
    authorized_users = [u for u in authorized_users if u['id'] != tg_id]
    save_authorized()
    await update.message.reply_text(f"➖ Removed {tg_id} from authorized users. ➖")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_owner(user_id):
        await update.message.reply_text("⚠️ you are not an admin ⚠️")
        return
    if not authorized_users:
        await update.message.reply_text("❌ No authorized users. ❌")
        return
    msg = "📜 Authorized users: 📜\n"
    for i, u in enumerate(authorized_users, 1):
        if u['id'] == OWNER_TG_ID:
            msg += f"{i}.(tg id {u['id']}) owner\n"
        elif u['username']:
            msg += f"{i}.(tg id {u['id']}) @{u['username']}\n"
        else:
            msg += f"{i}.(tg id {u['id']})\n"
    await update.message.reply_text(msg)

def main_bot():
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30)
    application = Application.builder().token(BOT_TOKEN).request(request).build()
    global APP, LOOP
    APP = application
    LOOP = asyncio.get_event_loop()
    
    # Restore tasks
    restore_tasks_on_start()
    
    # Start switch monitor
    monitor_thread = threading.Thread(target=switch_monitor, daemon=True)
    monitor_thread.start()
    
    # Post init for notifications
    async def post_init(app):
        for user_id, tasks_list in list(users_tasks.items()):
            for task in tasks_list:
                if task.get('type') == 'message_attack' and task['status'] == 'running':
                    await send_resume_notification(user_id, task)
    
    application.post_init = post_init

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("viewmyac", viewmyac))
    application.add_handler(CommandHandler("setig", setig))
    application.add_handler(CommandHandler("pair", pair_command))
    application.add_handler(CommandHandler("unpair", unpair_command))
    application.add_handler(CommandHandler("switch", switch_command))
    application.add_handler(CommandHandler("threads", threads_command))
    application.add_handler(CommandHandler("viewpref", viewpref))
    application.add_handler(CommandHandler("engage", engage_command))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("task", task_command))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("users", list_users))
    application.add_handler(CommandHandler("logout", logout_command))
    application.add_handler(CommandHandler("kill", cmd_kill))
    application.add_handler(CommandHandler("flush", flush))
    application.add_handler(CommandHandler("usg", usg_command))
    application.add_handler(CommandHandler("cancel", cancel_handler))

    conv_login = ConversationHandler(
        entry_points=[CommandHandler("login", login_start)],
        states={
            USERNAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_username)],
            PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_password)],
        },
        fallbacks=[],
    )
    application.add_handler(conv_login)

    conv_plogin = ConversationHandler(
        entry_points=[CommandHandler("plogin", plogin_start)],
        states={
            PLO_USERNAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, plogin_get_username)],
            PLO_PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, plogin_get_password)],
        },
        fallbacks=[],
    )
    application.add_handler(conv_plogin)

    conv_slogin = ConversationHandler(
        entry_points=[CommandHandler("slogin", slogin_start)],
        states={
            SLOG_SESSION: [MessageHandler(filters.TEXT & ~filters.COMMAND, slogin_get_session)],
            SLOG_USERNAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, slogin_get_username)],
        },
        fallbacks=[],
    )
    application.add_handler(conv_slogin)

    conv_attack = ConversationHandler(
        entry_points=[CommandHandler("attack", attack_start)],
        states={
            MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_mode)],
            SELECT_GC: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_gc_handler)],
            TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_target_handler)],
            MESSAGES: [
                MessageHandler(filters.Document.FileExtension("txt"), get_messages_file),
                MessageHandler(filters.TEXT & ~filters.COMMAND, get_messages),
            ],
        },
        fallbacks=[],
    )
    application.add_handler(conv_attack)

    # Conversation handler for engage text prompt
    conv_engage = ConversationHandler(
        entry_points=[CommandHandler("engage", engage_command)],
        states={
            ENGAGE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, engage_text_handler)],
        },
        fallbacks=[],
    )
    application.add_handler(conv_engage)

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🚀 Bot starting with message attack system!")
    application.run_polling()

if __name__ == "__main__":
    main_bot()