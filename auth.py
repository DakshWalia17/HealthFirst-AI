from fastapi import APIRouter
from pydantic import BaseModel
import json
import os


router = APIRouter()
USERS_FILE = "users.json"


if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump([], f)

class UserSignup(BaseModel):
    username: str
    password: str
    role: str 
class UserLogin(BaseModel):
    username: str
    password: str


@router.post("/signup/")
async def signup(user: UserSignup):
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    
    
    for u in users:
        if u["username"] == user.username:
            return {"status": "error", "message": "Already Exists"}
            
    
    users.append(user.dict())
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
        
    return {"status": "success", "message": f"Account created for {user.username} as {user.role}!"}

@router.post("/login/")
async def login(user: UserLogin):
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
        

    for u in users:
        if u["username"] == user.username and u["password"] == user.password:
            return {"status": "success", "role": u["role"], "message": "Login successful!"}
            
    return {"status": "error", "message": "Wrong username ya password!"}