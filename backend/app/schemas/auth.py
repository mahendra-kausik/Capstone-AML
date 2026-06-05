from pydantic import BaseModel, EmailStr, Field


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = "analyst"


class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str
