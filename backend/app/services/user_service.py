from typing import List, Optional
from datetime import datetime

from app.models.user import User, UserCreate, UserUpdate, UserStatus


class UserService:
    
    async def create_user(self, user_data: UserCreate, hashed_password: str) -> User:
        """Create a new user"""
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=user_data.role
        )
        await user.insert()
        return user

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return await User.find_one(User.username == username)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return await User.find_one(User.email == email)

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await User.get(user_id)

    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List all users"""
        return await User.find().skip(skip).limit(limit).to_list()

    async def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[User]:
        """Update user information"""
        user = await User.get(user_id)
        if not user:
            return None

        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()
        await user.save()
        return user

    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        user = await User.get(user_id)
        if not user:
            return False

        user.status = UserStatus.INACTIVE
        user.updated_at = datetime.utcnow()
        await user.save()
        return True

    async def activate_user(self, user_id: str) -> bool:
        """Activate a user"""
        user = await User.get(user_id)
        if not user:
            return False

        user.status = UserStatus.ACTIVE
        user.updated_at = datetime.utcnow()
        await user.save()
        return True

    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        user = await User.get(user_id)
        if not user:
            return False

        user.last_login = datetime.utcnow()
        await user.save()
        return True
