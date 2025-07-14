import base64

# Encode .pptx to Base64
def encode_pptx_to_base64(input_path, output_path):
    with open(input_path, "rb") as file:
        encoded = base64.b64encode(file.read())
    with open(output_path, "wb") as out_file:
        out_file.write(encoded)
    print(f"Encoded Base64 saved to: {output_path}")

# Decode Base64 back to .pptx
def decode_base64_to_pptx(input_path, output_path):
    with open(input_path, "rb") as file:
        decoded = base64.b64decode(file.read())
    with open(output_path, "wb") as out_file:
        out_file.write(decoded)
    print(f"Decoded .pptx saved to: {output_path}")

# Example usage
# encode_pptx_to_base64("presentation.pptx", "presentation.pptx.b64")
decode_base64_to_pptx("presentation.pptx.b64", "restored_presentation.pptx")
