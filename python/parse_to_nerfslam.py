import json
import sys
import numpy as np
import numpy.linalg as npl

def parse_numbers(line):
    # Split the line by spaces and convert each element to float
    return [float(num) for num in line.split()]

def create_file_path(index):
    return f"images/{index}"

def extract_rotation_translation(Rt):
    hh = np.eye(4)
    # 提取旋转矩阵
    hh[:3, :3] = Rt[:3, :3].T  # 取前3行前3列并转置
    # 提取平移向量
    hh[:3, 3] = hh[:3, :3] @ Rt[:3, 3]  # 使用旋转矩阵反向计算平移向量
    return hh
def create_depth_path(index):
    return f"images/{index}.depth.png"
def main():
    # Open the input text file
    with open(sys.argv[1], "r") as file:
        lines = file.readlines()

    # Initialize an empty list to store the JSON objects

    store = {
        "fl_x": 430,
        "fl_y": 430,
        "cx": 320.0,
        "cy": 260,
        "w": 640,
        "h": 512,
        "integer_depth_scale": 4.5777065690089265e-05,
        "aabb": [
            [
                -1.2991087436676025,
                -1.2797609567642212,
                -1.2809956073760986
            ],
            [
                1.2625601291656494,
                1.2819079160690308,
                1.2806732654571533
            ]
        ],
        "objects": [
            {
                "name": "book",
                "X_WO": [
                    [
                        0.8898758292198181,
                        0.45620277523994446,
                        8.670081115269568e-06,
                        -0.23871588706970215
                    ],
                    [
                        -0.4562027156352997,
                        0.8898757696151733,
                        -0.0004192583728581667,
                        -0.054532721638679504
                    ],
                    [
                        -0.00019898213213309646,
                        0.0003691325837280601,
                        0.9999999403953552,
                        -0.517119824886322
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "bound_box": [
                    [
                        -0.3512895107269287,
                        -0.17936690151691437,
                        -0.5195072889328003
                    ],
                    [
                        -0.1261833906173706,
                        0.07084791362285614,
                        -0.46570703387260437
                    ]
                ],
                "instance_id": 255
            },
            {
                "name": "laptop",
                "X_WO": [
                    [
                        0.9999999403953552,
                        5.258828616983794e-15,
                        0.0002685922954697162,
                        0.010611563920974731
                    ],
                    [
                        3.442735163616817e-08,
                        1.0000001192092896,
                        -0.0001281769946217537,
                        -0.11357051879167557
                    ],
                    [
                        -0.0002685922954697162,
                        0.00012817702372558415,
                        0.9999999403953552,
                        -0.5171721577644348
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "bound_box": [
                    [
                        -0.1607600748538971,
                        -0.2315729260444641,
                        -0.5255916118621826
                    ],
                    [
                        0.17550702393054962,
                        0.07014575600624084,
                        -0.30668771266937256
                    ]
                ],
                "instance_id": 200
            },
            {
                "name": "cup",
                "X_WO": [
                    [
                        0.9999999403953552,
                        5.258828616983794e-15,
                        0.0002685922954697162,
                        0.2725476324558258
                    ],
                    [
                        3.442735163616817e-08,
                        1.0000001192092896,
                        -0.0001281769946217537,
                        -0.1425350308418274
                    ],
                    [
                        -0.0002685922954697162,
                        0.00012817702372558415,
                        0.9999999403953552,
                        -0.5211877822875977
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "bound_box": [
                    [
                        0.18848299980163574,
                        -0.21760129928588867,
                        -0.5295241475105286
                    ],
                    [
                        0.3546352982521057,
                        -0.07653986662626266,
                        -0.4373883306980133
                    ]
                ],
                "instance_id": 150
            },
            {
                "name": "bluebell",
                "X_WO": [
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.2235679030418396
                    ],
                    [
                        0.0,
                        1.0,
                        0.0,
                        -0.019527241587638855
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        -0.41710442304611206
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "bound_box": [
                    [
                        0.1372528076171875,
                        -0.09432686120271683,
                        -0.5165746808052063
                    ],
                    [
                        0.2967168688774109,
                        0.049300335347652435,
                        -0.2910118103027344
                    ]
                ],
                "instance_id": 100
            }
        ]
    }

    json_output = []

    Tic = np.array([-0.00113207, -0.0158688, 0.999873,0.050166, -0.9999999, -0.000486594, -0.00113994,0.0474116, 0.000504622, -0.999874, -0.0158682,-0.0312415 ,0,0,0,1]).reshape((4, 4))

    
    # Iterate over each line
    for index, line in enumerate(lines, start=2):  # Start from '2' based on the example data
        # Extract the image name and the numbers
        parts = line.strip().split(maxsplit=1)
        image_name = parts[0]
        numbers = parse_numbers(parts[1])

        # Create the transform matrix
        transform_matrix = np.array([
            numbers[i:i+4] for i in range(0, len(numbers), 4)
        ])

        # transform_matrix = npl.inv(Tic) @ transform_matrix
        transform_matrix = extract_rotation_translation(transform_matrix)

        # Create the JSON object
        entry = {
            "transform_matrix": transform_matrix.tolist(),
            "file_path": create_file_path(index),
            "depth_path": create_depth_path(index)
        }

        # Append the JSON object to the list
        json_output.append(entry)

    store.update({
        "frames": json_output
    })
    # Write the JSON output to a file
    with open(sys.argv[2], "w") as outfile:
        json.dump(store, outfile, indent=4)

if __name__ == "__main__":
    main()
