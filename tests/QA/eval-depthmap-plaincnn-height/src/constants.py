from pathlib import Path

REPO_DIR = Path(__file__).parents[3].absolute()

'''
CODE_TO_SCANTYPE = {
    '100': '_front',
    '101': '_360',
    '102': '_back',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}

# Error margin on various ranges
#EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
EVALUATION_ACCURACIES = [.2, .4, .6, 1,  1.2, 2., 2.5, 3., 4., 5., 6.]

COLUMNS = ['qrcode', 'artifact', 'scantype', 'GT', 'predicted']
'''