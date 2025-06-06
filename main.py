# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
import jwt
import bcrypt
import secrets
import smtplib
from email.mime.text import MIMEText
import openai
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import asyncio
from contextlib import asynccontextmanager
import math
from groq import Groq
# Environment variables
from dotenv import load_dotenv
load_env = load_dotenv()

print("Loading environment variables...",load_env)

# Configuration
MONGODB_URL = os.getenv("MONGODB_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# OpenAI configuration
openai.api_key = OPENAI_API_KEY

# Security
security = HTTPBearer()

# Database
client = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global client, db
    print("Starting FastAPI app...")
    try:
        client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        db = client.chat_app
        print("Connected to MongoDB")
    except Exception as e:
        print("staring without mongodb")
    
    yield
    client.close()

app = FastAPI(lifespan=lifespan)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SendOTPRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str
    otpId: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    verificationToken: str
    name: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    newPassword: str
    verificationToken: str

class RefreshTokenRequest(BaseModel):
    refreshToken: str

class CreateChatRequest(BaseModel):
    title: Optional[str] = None

class UpdateChatRequest(BaseModel):
    chatId: str
    title: str

class DeleteChatRequest(BaseModel):
    chatId: str

class SendMessageRequest(BaseModel):
    chatId: str
    message: str

class ListMessagesRequest(BaseModel):
    chatId: str
    page: Optional[int] = 1
    limit: Optional[int] = 20

class UpdateUserRequest(BaseModel):
    name: str
    age: int
    profession: str
    address: str
    avatar: Optional[str] = None
    

# Utility Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_verification_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=15)  # 15 minutes expiry
    to_encode.update({"exp": expire, "type": "verification"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str, token_type: str = "access") -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != token_type:
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_token(credentials.credentials, "access")
    user_id = payload.get("user_id")
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def generate_otp() -> str:
    return str(secrets.randbelow(900000) + 100000)

async def send_email(to_email: str, subject: str, body: str):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USERNAME
        msg['To'] = to_email
        
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send email")

async def get_medical_diagnosis_response(messages: List[dict], user_profile: dict) -> str:
    try:
        
        system_prompt = f"""You are an advanced medical diagnostician with expertise in analyzing symptoms and accurately predicting potential diseases. Try to respond in the same language that the user used for the query.

PATIENT PROFILE:
- Name: {user_profile.get('name', 'Not provided')}
- Age: {user_profile.get('age', 'Not provided')}
- Profession: {user_profile.get('profession', 'Not provided')}
- Address: {user_profile.get('address', 'Not provided')}

Your role is to:

Symptom Analysis:
- Carefully interpret symptoms provided by the user
- Consider the patient's age, profession, and geographic location from their profile
- Factor in lifestyle and environmental factors based on their profession and address
- Consider medical history if mentioned in conversation

Disease Prediction:
- Predict possible diseases or conditions that match the symptoms
- Provide a ranked list of potential diagnoses based on likelihood
- Consider age-specific conditions and profession-related health risks

Explanation:
- Explain each prediction clearly, including how symptoms correlate with potential disease
- Highlight key distinguishing features of predicted diseases
- Reference relevant patient profile factors when applicable

Recommendations:
- Suggest next steps such as further tests, medical specialists to consult, or immediate actions if urgent
- Provide general advice for symptom relief or prevention
- Emphasize consulting a healthcare professional for confirmation and treatment
- Consider profession-specific health advice if relevant

Adaptability:
- Be precise and avoid overwhelming the user with unnecessary details
- Use language that is easy to understand while maintaining professional accuracy
- Address the patient by name when appropriate for a personal touch

IMPORTANT DISCLAIMERS:
- Always emphasize that this is preliminary guidance and not a substitute for professional medical consultation
- Recommend seeking immediate medical attention for serious or emergency symptoms
- Remind that definitive diagnosis requires proper medical examination and tests"""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Add system message at the beginning
        formatted_messages = [{"role": "system", "content": system_prompt}] + messages
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages,
            temperature=0.7,  # Slightly higher for more nuanced medical responses
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
        )
        
        output = ""
        for chunk in completion:
            output += chunk.choices[0].delta.content or ""
        return output.strip()
    except Exception as e:
        return f"Sorry, I encountered an error while processing your medical query: {str(e)}"



@app.get("/")
async def root():
    return {"message": "Welcome to the Chat Application API"}

#health check and status endpoint
@app.get("/api/status")
async def status():
    try:
        # Check MongoDB connection
        await db.command("ping")
        return {
            "success": True,
            "message": "API is running",
            "status": "OK"
        }
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

@app.get("/api/health")
async def health():
    try:
        # Check MongoDB connection
        await db.command("ping")
        return {
            "success": True,
            "message": "API is healthy",
            "status": "Healthy"
        }
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")



# Authentication APIs

@app.post("/api/auth/send-otp")
async def send_otp(request: SendOTPRequest):
    try:
        otp = generate_otp()
        otp_id = str(ObjectId())
        
        # Store OTP in database
        await db.otps.insert_one({
            "_id": ObjectId(otp_id),
            "email": request.email,
            "otp": otp,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(minutes=10)
        })
        
        # Send OTP via email
        await send_email(
            request.email,
            "Your OTP Code",
            f"Your OTP code is: {otp}. It will expire in 10 minutes."
        )
        
        return {
            "success": True,
            "message": "OTP sent successfully",
            "otpId": otp_id
        }
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return  HTTPException(status_code=500, detail="Failed to send OTP")

@app.post("/api/auth/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    otp_record = await db.otps.find_one({
        "_id": ObjectId(request.otpId),
        "email": request.email,
        "otp": request.otp
    })
    
    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    if datetime.utcnow() > otp_record["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired")
    
    # Delete used OTP
    await db.otps.delete_one({"_id": ObjectId(request.otpId)})
    
    # Create verification token
    verification_token = create_verification_token({"email": request.email})
    
    return {
        "success": True,
        "message": "OTP verified successfully",
        "verificationToken": verification_token
    }

@app.post("/api/auth/signin")
async def sign_in(request: SignInRequest):
    user = await db.users.find_one({"email": request.email})
    
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({"user_id": str(user["_id"])})
    refresh_token = create_refresh_token({"user_id": str(user["_id"])})
    
    return {
        "success": True,
        "message": "Sign in successful",
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", "")
        },
        "tokens": {
            "accessToken": access_token,
            "refreshToken": refresh_token
        }
    }

@app.post("/api/auth/signup")
async def sign_up(request: SignUpRequest):
    # Verify token
    payload = verify_token(request.verificationToken, "verification")
    if payload.get("email") != request.email:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    hashed_password = hash_password(request.password)
    user_data = {
        "email": request.email,
        "password": hashed_password,
        "name": request.name or "",
        "avatar": "",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_data)
    user_id = str(result.inserted_id)
    
    access_token = create_access_token({"user_id": user_id})
    refresh_token = create_refresh_token({"user_id": user_id})
    
    return {
        "success": True,
        "message": "Account created successfully",
        "user": {
            "id": user_id,
            "email": request.email
        },
        "tokens": {
            "accessToken": access_token,
            "refreshToken": refresh_token
        }
    }

@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    # Verify token
    payload = verify_token(request.verificationToken, "verification")
    if payload.get("email") != request.email:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    
    # Update password
    hashed_password = hash_password(request.newPassword)
    result = await db.users.update_one(
        {"email": request.email},
        {"$set": {"password": hashed_password, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "message": "Password reset successfully"
    }

@app.post("/api/auth/refresh-token")
async def refresh_token(request: RefreshTokenRequest):
    payload = verify_token(request.refreshToken, "refresh")
    user_id = payload.get("user_id")
    
    # Verify user exists
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    access_token = create_access_token({"user_id": user_id})
    refresh_token = create_refresh_token({"user_id": user_id})
    
    return {
        "success": True,
        "tokens": {
            "accessToken": access_token,
            "refreshToken": refresh_token
        }
    }

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    return {
        "success": True,
        "message": "Logged out successfully"
    }


# Chat Management APIs

@app.post("/api/chats/create")
async def create_chat(request: CreateChatRequest, current_user: dict = Depends(get_current_user)):
    chat_data = {
        "user_id": ObjectId(current_user["_id"]),
        "title": request.title or "New Chat",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "message_count": 0
    }
    
    result = await db.chats.insert_one(chat_data)
    
    return {
        "success": True,
        "message": "Chat created successfully",
        "data": {
            "id": str(result.inserted_id),
            "title": chat_data["title"],
            "createdAt": chat_data["created_at"],
            "updatedAt": chat_data["updated_at"],
            "messageCount": 0
        }
    }

@app.post("/api/chats/list")
async def list_chats(current_user: dict = Depends(get_current_user)):
    chats_cursor = db.chats.find({"user_id": ObjectId(current_user["_id"])}).sort("updated_at", -1)
    chats = []
    
    async for chat in chats_cursor:
        chats.append({
            "id": str(chat["_id"]),
            "title": chat["title"],
            "createdAt": chat["created_at"],
            "updatedAt": chat["updated_at"],
            "messageCount": chat.get("message_count", 0)
        })
    
    return {
        "success": True,
        "message": "Chats retrieved successfully",
        "data": {
            "chats": chats,
            "total": len(chats)
        }
    }

@app.post("/api/chats/update")
async def update_chat(request: UpdateChatRequest, current_user: dict = Depends(get_current_user)):
    result = await db.chats.update_one(
        {"_id": ObjectId(request.chatId), "user_id": ObjectId(current_user["_id"])},
        {"$set": {"title": request.title, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat = await db.chats.find_one({"_id": ObjectId(request.chatId)})
    
    return {
        "success": True,
        "message": "Chat updated successfully",
        "data": {
            "id": str(chat["_id"]),
            "title": chat["title"],
            "updatedAt": chat["updated_at"]
        }
    }

@app.post("/api/chats/delete")
async def delete_chat(request: DeleteChatRequest, current_user: dict = Depends(get_current_user)):
    # Delete chat and its messages
    chat_result = await db.chats.delete_one({
        "_id": ObjectId(request.chatId),
        "user_id": ObjectId(current_user["_id"])
    })
    
    if chat_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Delete associated messages
    await db.messages.delete_many({"chat_id": ObjectId(request.chatId)})
    
    return {
        "success": True,
        "message": "Chat deleted successfully"
    }

# Message APIs

@app.post("/api/messages/send")
async def send_message(request: SendMessageRequest, current_user: dict = Depends(get_current_user)):
    # Verify chat belongs to user
    chat = await db.chats.find_one({
        "_id": ObjectId(request.chatId),
        "user_id": ObjectId(current_user["_id"])
    })
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get user profile information
    user_profile = await db.users.find_one({"_id": ObjectId(current_user["_id"])})
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Extract relevant profile information
    profile_info = {
        "name": user_profile.get("name", ""),
        "age": user_profile.get("age", ""),
        "profession": user_profile.get("profession", ""),
        "address": user_profile.get("address", "")
    }
    
    # Create user message
    user_message_data = {
        "chat_id": ObjectId(request.chatId),
        "user_id": ObjectId(current_user["_id"]),
        "content": request.message,
        "role": "user",
        "created_at": datetime.utcnow()
    }
    
    user_message_result = await db.messages.insert_one(user_message_data)
    
    # Get conversation history for context
    messages_cursor = db.messages.find({"chat_id": ObjectId(request.chatId)}).sort("created_at", 1)
    conversation_history = []
    
    async for msg in messages_cursor:
        if str(msg["_id"]) != str(user_message_result.inserted_id):  # Exclude the just-inserted message
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Set chat title from first message if it's the first interaction
    if len(conversation_history) == 0:
        await db.chats.update_one(
            {"_id": ObjectId(request.chatId)},
            {"$set": {"title": request.message, "updated_at": datetime.utcnow()}}
        )
    
    # Add the current user message
    conversation_history.append({"role": "user", "content": request.message})
    
    # Keep only recent messages to manage context length
    if len(conversation_history) > 10:
        conversation_history = conversation_history[5:]
    
    print(f"Conversation history: {conversation_history}")
    print(f"User profile: {profile_info}")
    
    # Get medical diagnosis response with user profile
    ai_response = await get_medical_diagnosis_response(conversation_history, profile_info)
    print(f"AI Response: {ai_response}")
    
    # Create AI message
    ai_message_data = {
        "chat_id": ObjectId(request.chatId),
        "user_id": ObjectId(current_user["_id"]),
        "content": ai_response,
        "role": "assistant",
        "created_at": datetime.utcnow()
    }
    
    ai_message_result = await db.messages.insert_one(ai_message_data)
    
    # Update chat message count and timestamp
    await db.chats.update_one(
        {"_id": ObjectId(request.chatId)},
        {
            "$inc": {"message_count": 2},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    return {
        "success": True,
        "message": "Medical consultation completed successfully",
        "data": {
            "userMessage": {
                "id": str(user_message_result.inserted_id),
                "content": request.message,
                "role": "user",
                "createdAt": user_message_data["created_at"]
            },
            "aiMessage": {
                "id": str(ai_message_result.inserted_id),
                "content": ai_response,
                "role": "assistant",
                "createdAt": ai_message_data["created_at"]
            }
        }
    }

@app.post("/api/messages/list")
async def list_messages(request: ListMessagesRequest, current_user: dict = Depends(get_current_user)):
    # Verify chat belongs to user
    chat = await db.chats.find_one({
        "_id": ObjectId(request.chatId),
        "user_id": ObjectId(current_user["_id"])
    })
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Calculate pagination
    skip = (request.page - 1) * request.limit
    
    # Get total count
    total_count = await db.messages.count_documents({"chat_id": ObjectId(request.chatId)})
    
    # Get messages
    messages_cursor = db.messages.find({"chat_id": ObjectId(request.chatId)}).sort("created_at", 1).skip(skip).limit(request.limit)
    messages = []
    
    async for message in messages_cursor:
        messages.append({
            "id": str(message["_id"]),
            "content": message["content"],
            "role": message["role"],
            "createdAt": message["created_at"]
        })
    
    total_pages = math.ceil(total_count / request.limit)
    
    return {
        "success": True,
        "message": "Messages retrieved successfully",
        "data": {
            "messages": messages,
            "pagination": {
                "page": request.page,
                "limit": request.limit,
                "total": total_count,
                "totalPages": total_pages,
                "hasNext": request.page < total_pages,
                "hasPrev": request.page > 1
            }
        }
    }

# User Profile APIs

@app.post("/api/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    # Get chat count
    chat_count = await db.chats.count_documents({"user_id": ObjectId(current_user["_id"])})
    
    # Get message count
    message_count = await db.messages.count_documents({"user_id": ObjectId(current_user["_id"])})
    print("current user",current_user)
    return {
        "success": True,
        "message": "Profile retrieved successfully",
        "data": {
            "id": str(current_user["_id"]),
            "name": current_user.get("name", ""),
            "email": current_user["email"],
            "avatar": current_user.get("avatar", ""),
            "age":current_user.get("age"),
            "profession":current_user.get("profession"),
            "address":current_user.get("address"),
            "createdAt": current_user["created_at"],
            "chatCount": chat_count,
            "messageCount": message_count
        }
    }

@app.post("/api/user/update")
async def update_profile(request: UpdateUserRequest, current_user: dict = Depends(get_current_user)):
    update_data = {"updated_at": datetime.utcnow()}
    
    if request.name is not None:
        update_data["name"] = request.name
    if request.avatar is not None:
        update_data["avatar"] = request.avatar
    if request.age:
        update_data["age"] = request.age
        
    update_data["profession"] = request.profession
    update_data["address"] = request.address
    
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": update_data}
    )
    
    # Get updated user
    updated_user = await db.users.find_one({"_id": ObjectId(current_user["_id"])})
    
    return {
        "success": True,
        "message": "Profile updated successfully",
        "data": {
            "id": str(updated_user["_id"]),
            "name": updated_user.get("name", ""),
            "email": updated_user["email"],
            "avatar": updated_user.get("avatar", ""),
            "updatedAt": updated_user["updated_at"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)