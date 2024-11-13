import re

# Define the file path
file_path = '/home/ubuntu/venvs/sri-map-synth/lib/python3.12/site-packages/streamlit/web/server/server.py'

# Load the file and make the modification
with open(file_path, "r") as file:
    content = file.read()

# Use regex to match any integer value after "websocket_ping_timeout" and replace it with 3000
new_content = re.sub(
    r'("websocket_ping_timeout":\s*)\d+', 
    r'\1 300', 
    content
)

# Write the modified content back to the file
with open(file_path, "w") as file:
    file.write(new_content)

print("Value updated successfully.")