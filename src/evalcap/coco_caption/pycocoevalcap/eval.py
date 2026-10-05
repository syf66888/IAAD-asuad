__author__ = 'tylin'
from builtins import dict
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice

class COCOEvalCap:
    def __init__(self, coco, cocoRes, df, metrics=None):
        self.evalImgs = []
        self.eval = dict()
        self.imgToEval = dict()
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

        self.gts = None
        self.res = None
        self.df = df
        self.metrics = set(metrics) if metrics is not None else {'Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'}
        unknown = self.metrics - {'Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'}
        if unknown:
            raise ValueError('Unknown caption metrics: %s' % sorted(unknown))

    def tokenize(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = dict()
        res = dict()
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        self.gts  = tokenizer.tokenize(gts)
        self.res = tokenizer.tokenize(res)

    def evaluate(self):
        self.tokenize()

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        # Construct only requested scorers: METEOR/SPICE start external Java
        # processes and are unnecessary for B4/CIDEr/ROUGE-L experiments.
        scorers = []
        if 'Bleu' in self.metrics:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in self.metrics:
            scorers.append((Meteor(), 'METEOR'))
        if 'ROUGE_L' in self.metrics:
            scorers.append((Rouge(), 'ROUGE_L'))
        if 'CIDEr' in self.metrics:
            scorers.append((Cider(self.df), 'CIDEr'))
        if 'SPICE' in self.metrics:
            scorers.append((Spice(), 'SPICE'))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, self.gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, self.gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
