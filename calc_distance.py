import math

def calculate_euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

if __name__ == "__main__":
    [0, 1, 25, 31, 7, 26]
    distance1 = calculate_euclidian_distance(10.5, 14.4, 11.2, 14.1)
    distance2 = calculate_euclidian_distance(11.2, 14.1, 7.4, 24)
    distance3 = calculate_euclidian_distance(7.4, 24, 7.3, 18.8)
    distance4 = calculate_euclidian_distance(7.3, 18.8, 16.3, 13.3)
    distance5 = calculate_euclidian_distance(16.3, 13.3, 8.2, 19.9)

    print(distance1 + distance2 + distance3 + distance4 + distance5)
    print(40 - (distance1 + distance2 + distance3 + distance4 + distance5))