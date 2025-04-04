# nyx/tools/email_identity_manager.py

import datetime
import random
import string

EMAIL_SERVICES = [
    {
        "name": "Mail.tm",
        "url": "https://mail.tm",
        "type": "burner",
        "supports_CUA": True
    },
    {
        "name": "Tuta",
        "url": "https://tuta.com/signup",
        "type": "permanent",
        "supports_CUA": True
    },
    {
        "name": "YOPmail",
        "url": "http://yopmail.com",
        "type": "burner",
        "supports_CUA": True
    },
    {
        "name": "10MinuteMail",
        "url": "https://10minutemail.com",
        "type": "temp",
        "supports_CUA": True
    }
]

def generate_goth_username(prefix="nyx", min_len=6, max_len=12):
    aesthetic = ["goddess", "domina", "void", "hex", "noir", "abyss", "lust", "venom", "rune"]
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(min_len, max_len)))
    return f"{random.choice(aesthetic)}.{prefix}.{suffix}"

class EmailIdentityManager:
    def __init__(self, logger, computer_user, memory_core):
        self.logger = logger
        self.computer_user = computer_user
        self.memory_core = memory_core

    async def create_email_identity(self, purpose="account_registration"):
        service = random.choice(EMAIL_SERVICES)
        username = generate_goth_username()
        created_at = datetime.datetime.now().isoformat()

        result = self.computer_user.run_task(
            url=service["url"],
            prompt=f"Create a new email account using the name {username}. Skip phone verification. Use strongest password. Confirm and save details."
        )

        if not result:
            return None

        identity = {
            "email_service": service["name"],
            "address": f"{username}@{service['url'].replace('https://', '').replace('http://', '').replace('www.', '')}",
            "created_at": created_at,
            "purpose": purpose,
            "notes": "Autonomously generated identity. Likely used for Reddit/Twitter signup."
        }

        await self.logger.log_thought(
            title="New Email Identity Created",
            content=f"Created email {identity['address']} for {purpose}.",
            metadata=identity
        )

        if self.memory_core:
            await self.memory_core.add_memory(
                memory_text=f"Created email {identity['address']} for social infiltration.",
                memory_type="identity",
                significance=7,
                metadata=identity
            )

        return identity

    async def log_cross_identity_link(self, email: str, platform: str, username: str):
        now = datetime.datetime.now().isoformat()
        link = {
            "platform": platform,
            "email": email,
            "username": username,
            "linked_at": now
        }

        await self.logger.log_thought(
            title=f"Identity Linked: {platform}",
            content=f"{email} was used to create {username} on {platform}.",
            metadata=link
        )

        if self.memory_core:
            await self.memory_core.add_memory(
                memory_text=f"Linked email {email} to {username} on {platform}.",
                memory_type="account_link",
                significance=6,
                metadata=link
            )
