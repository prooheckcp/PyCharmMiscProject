def convert_camera_values(x, y, z):
    return 10 + z, y, -x;

def convert_values(x, y, z):
    return -z, y, x - 10

def double_convert_values(x, y, z):
    x2, y2, z2 = convert_values(x, y, z)
    return convert_values(x2, y2, z2)

print("Rocksteady")
print(double_convert_values(10, 1, -10))