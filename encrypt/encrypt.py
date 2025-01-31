import os
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def generate_key():
    return get_random_bytes(32) 

def encrypt_file(file_path, key, output_folder):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv

    with open(file_path, 'rb') as file:
        plaintext = file.read()

    pad_len = 16 - (len(plaintext) % 16)
    plaintext += bytes([pad_len] * pad_len)

    relative_path = os.path.relpath(file_path, start=os.path.dirname(file_path))
    output_path = os.path.join(output_folder, relative_path + '.aes')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  

    with open(output_path, 'wb') as file_out:
        file_out.write(iv)  
        ciphertext = cipher.encrypt(plaintext)
        file_out.write(ciphertext)

def decrypt_file(file_path, key, output_folder):
    with open(file_path, 'rb') as file_in:
        iv = file_in.read(16)  
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = file_in.read()

    plaintext = cipher.decrypt(ciphertext)

    pad_len = plaintext[-1]
    plaintext = plaintext[:-pad_len]

    output_path = os.path.join(output_folder, os.path.basename(file_path).replace('.aes', ''))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  

    with open(output_path, 'wb') as file_out:
        file_out.write(plaintext)

def main():
    action = input("Enter 'e' for encryption or 'd' for decryption: ").strip().lower()
    
    folder_path = input("Enter the path to the folder with files: ").strip()
    output_folder = input("Enter the path to save (leave blank for the same folder): ").strip() or folder_path

    if action == 'e':
        key = generate_key()
        print(f"Generated key: {base64.b64encode(key).decode()}")

        try:
            for root, _, files in os.walk(folder_path):  
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if filename.endswith('.py'):  
                        encrypt_file(file_path, key, output_folder)
                        print(f"Encrypted: {file_path} -> {output_folder}")
        except Exception as e:
            print(f"Error: {e}")

    elif action == 'd':
        key_input = input("Enter the key for decryption: ")
        key = base64.b64decode(key_input)

        try:
            for root, _, files in os.walk(folder_path):  
                for filename in files:
                    if filename.endswith('.aes'):  
                        file_path = os.path.join(root, filename)
                        decrypt_file(file_path, key, output_folder)
                        print(f"Decrypted: {file_path} -> {output_folder}")
        except Exception as e:
            print(f"Error: {e}")

    else:
        print("Invalid action.")

if __name__ == "__main__":
    main()
