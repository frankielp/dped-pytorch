def get_specified_res(phone, resolution):
    default_res = default_resolutions()
    if resolution == "orig":
        IMAGE_HEIGHT = default_res[phone][0]
        IMAGE_WIDTH = default_res[phone][1]
    else:
        IMAGE_HEIGHT = default_res[resolution][0]
        IMAGE_WIDTH = default_res[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


# Support function
def default_resolutions():
    # IMAGE_HEIGHT, IMAGE_WIDTH

    default_res = {}
    # Based on phone
    default_res["iphone"] = [1536, 2048]
    default_res["iphone_orig"] = [1536, 2048]
    default_res["blackberry"] = [1560, 2080]
    default_res["blackberry_orig"] = [1560, 2080]
    default_res["sony"] = [1944, 2592]
    default_res["sony_orig"] = [1944, 2592]

    # Based on resolution
    default_res["high"] = [1260, 1680]
    default_res["medium"] = [1024, 1366]
    default_res["small"] = [768, 1024]
    default_res["tiny"] = [600, 800]

    return default_res


def extract_crop(image, resolution, phone):
    default_res = default_resolutions()

    if resolution == "orig":
        return image

    else:
        x_up = int((default_res[phone][1] - default_res[resolution][1]) / 2)
        y_up = int((default_res[phone][0] - default_res[resolution][0]) / 2)

        x_down = x_up + default_res[resolution][1]
        y_down = y_up + default_res[resolution][0]

        return image[y_up:y_down, x_up:x_down, :]
