import base64
import os

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# 32-byte AES-256 key
AES_KEY = b"1234567890ABCDEF1234567890ABCDEF"


def encrypt_data(data: str):
    """
    Encrypt a string using AES-256 (CBC mode).
    Returns (encrypted_text, iv)
    """

    iv = get_random_bytes(16)

    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)

    encrypted = cipher.encrypt(
        pad(data.encode(), AES.block_size)
    )

    encrypted_text = base64.b64encode(encrypted).decode()

    iv_text = base64.b64encode(iv).decode()

    return encrypted_text, iv_text


def decrypt_data(encrypted_text: str, iv_text: str):
    """
    Decrypt AES encrypted text.
    """

    encrypted = base64.b64decode(encrypted_text)

    iv = base64.b64decode(iv_text)

    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)

    decrypted = unpad(
        cipher.decrypt(encrypted),
        AES.block_size
    )

    return decrypted.decode()