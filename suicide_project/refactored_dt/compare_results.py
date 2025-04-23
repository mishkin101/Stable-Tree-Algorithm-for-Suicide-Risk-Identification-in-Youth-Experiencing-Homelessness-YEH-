from PIL import Image, ImageChops
from pathlib import Path

OUTPUT = Path("outputs/") 
OUTPUT_PATH = OUTPUT.resolve()

COMPARISION_PATH = OUTPUT/"comparisions/"
COMPARISION_PATH.mkdir(exist_ok=True)

PATH_OUTPUT_FROM_GIVEN_CODE = Path(OUTPUT/"from_given_code")

PLOT_NAMES_SUICATTEMPT = ['Decision_Tree_suicattempt.png', 'Feature_Importance_suicattempt.png', 'ROC_Curve_suicattempt.png']
PLOT_NAMES_SUICIDEA = ['Decision_Tree_suicidea.png', 'Feature_Importance_suicidea.png', 'ROC_Curve_suicidea.png']

METRICS_SUICATTEMPT = 'Metrics_suicattempt.png'
METRICS_SUICIDEA = 'Metrics_suicattempt.png'

# get difference in images for those in "output/from_given_code" and "output/"
def tree_difference_png(tree_img1_path, tree_img2_path):
    tree_img1 = Image.open(tree_img1_path)
    tree_img2 = Image.open(tree_img2_path)
    
    diff = ImageChops.difference(tree_img1, tree_img2)

    # if diff.getbbox():
    #     diff.show()
    return diff

def grid_difference_png(image_folder1_path, image_folder2_path=PATH_OUTPUT_FROM_GIVEN_CODE):
    image_folder1 = Path(image_folder1_path)
    image_folder2 = Path(image_folder2_path)
    
    # Compare the images in the two directories using a 2x3 grid of of images for the differences
    rows = 2
    cols = 3
    imgs = []
    
    for plot_name in PLOT_NAMES_SUICATTEMPT:
        img = tree_difference_png(image_folder1/plot_name, image_folder2/plot_name)
        imgs.append(img)
    for plot_name in PLOT_NAMES_SUICIDEA:
        img = tree_difference_png(image_folder1/plot_name, image_folder2/plot_name)
        imgs.append(img)
        
    grid_w, grid_h = imgs[0].size
    grid = Image.new('RGB', size=(cols*grid_w, rows*grid_h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*grid_w, i//cols*grid_h))
    grid.show()
    grid.save(OUTPUT_PATH/'grid_difference.png')
    # grid_w, grid_h = imgs[0].size
    # grid = Image.new('RGB', size=(cols*grid_w, rows*grid_h))
    # for i, img in enumerate(imgs):
    #     grid.paste(img, box=(i%cols*grid_w, i//cols*grid_h))
    # grid.show()
    
    return grid

if __name__ == '__main__':
    grid_difference_png(COMPARISION_PATH, PATH_OUTPUT_FROM_GIVEN_CODE)