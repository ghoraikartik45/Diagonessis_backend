# db_setup.py
"""
MongoDB Collections Schema and Indexes Setup
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def setup_database():
    """Setup MongoDB collections and indexes"""
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
    db = client.chat_app
    
    # Users Collection
    users_indexes = [
        IndexModel([("email", ASCENDING)], unique=True),
        IndexModel([("created_at", DESCENDING)])
    ]
    await db.users.create_indexes(users_indexes)
    print("✅ Users collection indexes created")
    
    # OTPs Collection
    otps_indexes = [
        IndexModel([("email", ASCENDING)]),
        IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),  # TTL index
        IndexModel([("created_at", DESCENDING)])
    ]
    await db.otps.create_indexes(otps_indexes)
    print("✅ OTPs collection indexes created")
    
    # Chats Collection
    chats_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        IndexModel([("created_at", DESCENDING)])
    ]
    await db.chats.create_indexes(chats_indexes)
    print("✅ Chats collection indexes created")
    
    # Messages Collection
    messages_indexes = [
        IndexModel([("chat_id", ASCENDING)]),
        IndexModel([("chat_id", ASCENDING), ("created_at", ASCENDING)]),
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)])
    ]
    await db.messages.create_indexes(messages_indexes)
    print("✅ Messages collection indexes created")
    
    client.close()
    print("🎉 Database setup completed!")

"""
MongoDB Collections Schema:

1. users
{
    "_id": ObjectId,
    "email": "user@example.com",
    "password": "hashed_password",
    "name": "John Doe",
    "avatar": "https://example.com/avatar.jpg",
    "created_at": ISODate,
    "updated_at": ISODate
}

2. otps
{
    "_id": ObjectId,
    "email": "user@example.com",
    "otp": "123456",
    "created_at": ISODate,
    "expires_at": ISODate  // TTL index will auto-delete expired OTPs
}

3. chats
{
    "_id": ObjectId,
    "user_id": ObjectId,
    "title": "Chat Title",
    "message_count": 0,
    "created_at": ISODate,
    "updated_at": ISODate
}

4. messages
{
    "_id": ObjectId,
    "chat_id": ObjectId,
    "user_id": ObjectId,
    "content": "Message content",
    "role": "user" | "assistant",
    "created_at": ISODate
}
"""

if __name__ == "__main__":
    asyncio.run(setup_database())