import os
import hashlib
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import bcrypt
from jose import jwt, JWTError


USERS_FILE = Path("users.json")

#JWT config
JWT_SECRET_KEY = os.environ.get("KWX_JWT_SECRET", "dev-fallback-not-for-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60


#password hashing (bcrypt)
def hash_password_bcrypt(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password_bcrypt(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


#JWT tokens
def create_token(user_id: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_user_id() -> str:
    return f"USR-{uuid.uuid4().hex[:12].upper()}"


def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)



def create_user(username: str, password: str, organization: str = None, verification_link: str = None) -> Tuple[bool, str]:
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    

    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    #create user
    user_id = generate_user_id()
    password_hash = hash_password_bcrypt(password)
    
    users[username] = {
        "user_id": user_id,
        "password_hash": password_hash,
        "hash_method": "bcrypt",
        "salt": "",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "active": True,
        "organization": organization,
        "verification_link": verification_link
    }
    save_users(users)
    
    return True, user_id


def authenticate(username: str, password: str) -> Tuple[bool, Optional[str]]:
    users = load_users()
    
    if username not in users:
        return False, None
    
    user = users[username]
    
    if not user.get("active", True):
        return False, None
    
    hash_method = user.get("hash_method", "sha256")
    
    if hash_method == "bcrypt":
        if verify_password_bcrypt(password, user["password_hash"]):
            return True, user["user_id"]
    
    return False, None

def get_user_id(username: str) -> Optional[str]:
    users = load_users()
    if username in users:
        return users[username]["user_id"]
    return None


def get_user_profile(username: str) -> Optional[dict]: #org link or org name
    users = load_users()
    if username not in users:
        return None
    user = users[username]
    return {
        "user_id": user["user_id"],
        "organization": user.get("organization"),
        "verification_link": user.get("verification_link")
    }


def get_profile_by_id(user_id: str) -> Optional[dict]: #get optional public profile using ID
    users = load_users()
    for username, data in users.items():
        if data["user_id"] == user_id:
            return {
                "user_id": user_id,
                "organization": data.get("organization"),
                "verification_link": data.get("verification_link")
            }
    return None


def deactivate_user(username: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    users[username]["active"] = False
    save_users(users)
    return True


def list_users() -> list:
    users = load_users()
    return [
        {"username": u, "user_id": data["user_id"], "active": data.get("active", True)}
        for u, data in users.items()
    ]



if __name__ == "__main__":
    import sys
    
    #usage of code
    if len(sys.argv) < 2:
        print("Kwiddex Secure Login")
        print("Usage:")
        print("  python auth.py create <username> <password>")
        print("  python auth.py login <username> <password>")
        print("  python auth.py list")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "create" and len(sys.argv) >= 4:
        username = sys.argv[2]
        password = sys.argv[3]
        org = sys.argv[4] if len(sys.argv) > 4 else None
        link = sys.argv[5] if len(sys.argv) > 5 else None
        success, result = create_user(username, password, org, link)
        if success:
            print(f"User created. ID: {result}")
        else:
            print(f"Error: {result}")
    
    elif command == "login" and len(sys.argv) >= 4:
        username = sys.argv[2]
        password = sys.argv[3]
        success, user_id = authenticate(username, password)
        if success:
            print(f"Login successful")
            print(f"User ID: {user_id}")
            print(f"Token: {token}")

        else:
            print("Invalid username or password")
    
    elif command == "list":
        users = list_users()
        if not users:
            print("No users found")
        else:
            print("Users:")
            for u in users:
                status = "active" if u["active"] else "inactive"
                print(f"  {u['username']} ({u['user_id']}) - {status}")
    
    else:
        print("Invalid command. Run without arguments for help.")
