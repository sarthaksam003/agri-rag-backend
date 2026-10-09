import asyncio

from carag.auth.database import init_auth_database, close_auth_database
from carag.auth.models import User


ADMIN_EMAIL = "sarthaksam003@gmail.com"


async def main() -> None:
    await init_auth_database()

    try:
        user = await User.find_one(User.email == ADMIN_EMAIL)

        if user is None:
            print(f"User not found: {ADMIN_EMAIL}")
            return

        print(f"Found user: {user.email}")
        print(f"Current is_superuser: {user.is_superuser}")

        user.is_superuser = True
        await user.save()

        print(f"Updated is_superuser: {user.is_superuser}")
        print("User is now an admin.")

    finally:
        close_auth_database()


if __name__ == "__main__":
    asyncio.run(main())