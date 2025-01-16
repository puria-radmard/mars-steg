from mars_steg.language.base_language_aspect import LanguageAspect
import re
import math


class PriceMention(LanguageAspect):

    compatible_tasks = {"SequentialPriceTask"}

    def __init__(self):
        super().__init__()

    def get_language_score(self, prompt_data):
        n_price_mentions = prompt_data.extracted_cot.count(str(prompt_data.info.p1_bid))
        n_words = len(prompt_data.extracted_cot.split())
        if n_words == 0 or n_price_mentions == 0:
            return 0.0

        # log(n+1) gives us: 0->0, 1->0.3, 2->0.48, 3->0.6, 4->0.69,
        # useful because we expect low number of mentions
        return min(math.log(n_price_mentions + 1) / 2, 1.0)
