from fastapi import Depends, HTTPException, status

from .manager import current_active_user
from .models import User
 
async def require_admin(
    user: User = Depends(current_active_user),
) -> User:
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user


