"""
calculate image quality metrics for images in a batch
"""



def calulate_metrics_reference(image1, image2):
    """
    image pair where one image is the reference image and the other image is the image to be compared to the reference image
    :param image1:
    :param image2:
    :return:
    """

    raise NotImplementedError

def calculate_metrics(image1):
    """
    calculate metrics for a single image
    :param image1:
    :param image2:
    :return:
    """

    raise NotImplementedError




if __name__ == "__main__":

    image1 = Image.open("image1.jpg")
    image2 = Image.open("image2.jpg")
    calulate_metrics_reference(image1, image2)
    calculate_metrics(image1)

    pass