from typing import Dict, List, Optional, Tuple, Union

from emoji import UNICODE_EMOJI
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

# Make sure multi-character emoji don't contain whitespace
EMOJI = {e.replace(" ", ""): t for e, t in UNICODE_EMOJI.items()}

DEFAULT_ATTRS = ("has_emoji", "is_emoji", "emoji_desc", "emoji")

DEFAULT_CONFIG = {}


@Language.factory("twiiter", default_config=DEFAULT_CONFIG)
def create_twitter(
    nlp: Language,
    name: str,
    merge_spans: bool = True,
    lookup: Optional[Dict[str, str]] = None,
    pattern_id: str = "EMOJI",
    attrs: Tuple[str, str, str, str] = DEFAULT_ATTRS,
    force_extension: bool = True,
):
    return Twitter(nlp, merge_spans, lookup, pattern_id, attrs, force_extension)


class Twitter:
    """spaCy pipeline component for added hashtags, mentions and urls to
    `Doc`objects.

    Examples:
        >>> import spacy
        >>> from dacy.twitter import Twitter
        >>> nlp = spacy.load("en_core_web_sm")
        >>> nlp.add_pipe("twitter", first=True)
        >>> doc = nlp("@Charles life is great right www.greatlife.com! #yolo")
        >>> doc._.has_hashtag
        True
        >>> doc[2:5]._.has_hashtag
        True
        >>> doc[-1]._.is_hashtag
        True
        >>> doc[0]._.is_mention is True
        True
        >>> doc._.hashtags
        ["#yolo"]
        >>> doc._.hastag_indexes
        [(48, 53)]
        >>> doc[0]._.is_mention
        True
        >>> doc[6]._.is_url
        True
    """

    name = "twitter"

    def __init__(
        self,
        nlp: Language,
        redo_tokenization: bool = True,
        lookup: Optional[Dict[str, str]] = None,
        pattern_id: str = "EMOJI",
        attrs: Tuple[str, str, str, str] = DEFAULT_ATTRS,
        force_extension: bool = True,
    ) -> None:
        """Initialise the pipeline component.

        nlp (Language): The shared nlp object. Used to initialise the
        matcher     with the shared `Vocab`, and create `Doc` match
        patterns. attrs (tuple): Attributes to set on the ._ property.
        Defaults to     ('has_emoji', 'is_emoji', 'emoji_desc',
        'emoji'). pattern_id (unicode): ID of match pattern, defaults to
        'EMOJI'. Can be     changed to avoid ID clashes. merge_spans
        (bool): Merge spans containing multi-character emoji. Will
        only merge combined emoji resulting in one icon, not sequences.
        lookup (dict): Optional lookup table that maps emoji unicode
        strings     to custom descriptions, e.g. translations or other
        annotations. RETURNS (callable): A spaCy pipeline component.
        """
        self._has_emoji, self._is_emoji, self._emoji_desc, self._emoji = attrs
        self.merge_spans = merge_spans
        self.lookup = lookup or {}
        self.matcher = PhraseMatcher(nlp.vocab)
        emoji_patterns = list(nlp.tokenizer.pipe(EMOJI.keys()))
        self.matcher.add(pattern_id, None, *emoji_patterns)
        # Add attributes
        Doc.set_extension(self._has_emoji, getter=self.has_emoji, force=force_extension)
        Doc.set_extension(self._emoji, getter=self.iter_emoji, force=force_extension)
        Span.set_extension(
            self._has_emoji,
            getter=self.has_emoji,
            force=force_extension,
        )
        Span.set_extension(self._emoji, getter=self.iter_emoji, force=force_extension)
        Token.set_extension(self._is_emoji, default=False, force=force_extension)
        Token.set_extension(
            self._emoji_desc,
            getter=self.get_emoji_desc,
            force=force_extension,
        )

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipeline component to a `Doc` object.

        doc (Doc): The `Doc` returned by the previous pipeline
        component. RETURNS (Doc): The modified `Doc` object.
        """
        spans = self.matcher(doc, as_spans=True)
        for span in spans:
            for token in span:
                token._.set(self._is_emoji, True)

        if self.merge_spans:
            spans = filter_spans(spans)
            with doc.retokenize() as retokenizer:
                for span in spans:
                    if len(span) > 1:
                        retokenizer.merge(span)
        return doc

    def has_emoji(self, tokens: Union[Doc, Span]) -> bool:
        return any(token._.get(self._is_emoji) for token in tokens)

    def iter_emoji(self, tokens: Union[Doc, Span]) -> List[Tuple[str, int, str]]:
        return [
            (t.text, i, t._.get(self._emoji_desc))
            for i, t in enumerate(tokens)
            if t._.get(self._is_emoji)
        ]

    def get_emoji_desc(self, token: Token) -> Optional[str]:
        if token.text in self.lookup:
            return self.lookup[token.text]
        if token.text in EMOJI:
            desc = EMOJI[token.text]
            # Here we're converting shortcodes, e.g. ":man_getting_haircut:"
            return desc.replace("_", " ").replace(":", "")
        return None


def hashtag_getter(doc: Doc) -> List[str]:
    """Extract hashtags from text.

    Args:
        doc (Doc): A SpaCy document

    Returns:
        List[str]: A list of hashtags

    Example:
        >>> from spacy.tokens import Doc
        >>> Doc.set_extension("hashtag", getter=dacy.utilities.twitter.hashtags)
        >>> doc = nlp("Fuck hvor fedt! #yolo #life")
        >>> doc._.hashtag  # extrac the hashtags from your document
        ["#yolo", "#life"]
    """

    def find_hashtags(
        text,
        valid_tags={"#", "＃"},
        valid_chars={"_", "-"},
        invalid_tag_suffix={b"\xe2\x83\xa3", b"\xef\xb8\x8f"},
    ):
        def is_letter(t):
            if (
                t.isalnum()
                or t in valid_chars
                or str.encode(t).startswith(b"\xcc")
                or str.encode(t).startswith(b"\xe0")
            ):
                return True
            return False

        start = None
        for i, char in enumerate(text):
            if (
                char in valid_tags
                and not (
                    i + 1 != len(text) and str.encode(text[i + 1]) in invalid_tag_suffix
                )
                and (i == 0 or not (is_letter(text[i - 1]) or text[i - 1] == "&"))
            ):
                start = i
                continue
            if start is not None and not is_letter(char):
                if char in valid_tags:
                    start = None
                    continue
                print(start, i)
                if not text[start + 1 : i].isnumeric():
                    yield "#" + text[start + 1 : i]
                start = None
        if start is not None and not text[start + 1 : i + 1].isnumeric():
            print(start, i)
            yield "#" + text[start + 1 : i + 1]

    return list(find_hashtags(doc.text))


# def url_getter(doc: Doc) -> List[str]:
#     """
#     Extract hashtags from text

#     Args:
#         doc (Doc): A SpaCy document

#     Returns:
#         List[str]: A list of ulrs

#     Example:
#         >>> from spacy.tokens import Doc
#         >>> Doc.set_extension("hashtag", getter=dacy.utilities.twitter.hashtags)
#         >>> doc = nlp("Fuck hvor fedt! www.cool.dk")
#         >>> doc._.hashtag  # extrac the hashtags from your document
#         ["www.cool.dk"]
#     """
#     O = len(doc)
#     P = len(list(doc.sents))
#     L = len([t for t in doc if len(t) > 6])

#     LIX = O / P + L * 100 / O


#     return LIX


# def mentions_getter(doc: Doc) -> List[str]:
#     """
#     Extract hashtags from text

#     Args:
#         doc (Doc): A SpaCy document

#     Returns:
#         List[str]: A list of mentions

#     Example:
#         >>> from spacy.tokens import Doc
#         >>> Doc.set_extension("hashtag", getter=dacy.utilities.twitter.hashtags)
#         >>> doc = nlp("Fuck hvor fedt @arne!")
#         >>> doc._.hashtag  # extrac the hashtags from your document
#         ["@arne"]
#     """
#     O = len(doc)
#     P = len(list(doc.sents))
#     L = len([t for t in doc if len(t) > 6])

#     LIX = O / P + L * 100 / O


#     return LIX


def find_hashtags(
    text,
    valid_tags={"#", "＃"},
    valid_chars={"_", "-"},
    invalid_tag_suffix={b"\xe2\x83\xa3", b"\xef\xb8\x8f"},
):
    def is_letter(t):
        if (
            t.isalnum()
            or t in valid_chars
            or str.encode(t).startswith(b"\xcc")
            or str.encode(t).startswith(b"\xe0")
        ):
            return True
        return False

    start = None
    for i, char in enumerate(text):
        if (
            char in valid_tags
            and not (
                i + 1 != len(text) and str.encode(text[i + 1]) in invalid_tag_suffix
            )
            and (i == 0 or not (is_letter(text[i - 1]) or text[i - 1] == "&"))
        ):
            start = i
            continue
        if start is not None and not is_letter(char):
            if char in valid_tags:
                start = None
                continue
            print(start, i)
            if not text[start + 1 : i].isnumeric():
                yield "#" + text[start + 1 : i]
            start = None
    if start is not None and not text[start + 1 : i + 1].isnumeric():
        print(start, i)
        yield "#" + text[start + 1 : i + 1]


for s in [
    "Fuck hvor fedt! #yolo #life",
    "#taga",
    "invalid hashtags #double#tag",
    "invalid hashtags #double@tag",
    "also invalid #1",
]:
    result = list(find_hashtags(s))
    print(s, "\t", result)

# from spacy.lang.da import Danish
# nlp = Danish()
# doc = nlp("Fuck hvor fedt! #yolo #life")
# doc = nlp("invalid hashtags #double#tag")
# doc = nlp("invalid hashtags #double@trouble")
# city_getter = lambda doc: any(city in doc.text for city in ("New York", "Paris", "Berlin"))
# Doc.set_extension("has_city", getter=city_getter)


hash_tag_examples = """"
    - description: "Autolink trailing hashtag"
      text: "text #hashtag"
      expected: "text <a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">#hashtag</a>"

    - description: "Autolink alphanumeric hashtag (letter-number-letter)"
      text: "text #hash0tag"
      expected: "text <a href=\"https://twitter.com/search?q=%23hash0tag\" title=\"#hash0tag\" class=\"tweet-url hashtag\">#hash0tag</a>"

    - description: "Autolink alphanumeric hashtag (number-letter)"
      text: "text #1tag"
      expected: "text <a href=\"https://twitter.com/search?q=%231tag\" title=\"#1tag\" class=\"tweet-url hashtag\">#1tag</a>"

    - description: "Autolink hashtag with underscore"
      text: "text #hash_tag"
      expected: "text <a href=\"https://twitter.com/search?q=%23hash_tag\" title=\"#hash_tag\" class=\"tweet-url hashtag\">#hash_tag</a>"

    - description: "DO NOT Autolink all-numeric hashtags"
      text: "text #1234"
      expected: "text #1234"

    - description: "DO NOT Autolink hashtag preceded by a letter"
      text: "text#hashtag"
      expected: "text#hashtag"

    - description: "DO NOT Autolink hashtag that begins with \ufe0f (Emoji style hash sign)"
      text: "#️hashtag"
      expected: "#️hashtag"

    - description: "DO NOT Autolink hashtag that begins with \ufe0f (Keycap style hash sign)"
      text: "#⃣hashtag"
      expected: "#⃣hashtag"

    - description: "Autolink multiple hashtags"
      text: "text #hashtag1 #hashtag2"
      expected: "text <a href=\"https://twitter.com/search?q=%23hashtag1\" title=\"#hashtag1\" class=\"tweet-url hashtag\">#hashtag1</a> <a href=\"https://twitter.com/search?q=%23hashtag2\" title=\"#hashtag2\" class=\"tweet-url hashtag\">#hashtag2</a>"

    - description: "Autolink hashtag preceded by a period"
      text: "text.#hashtag"
      expected: "text.<a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">#hashtag</a>"

    - description: "DO NOT Autolink hashtag preceded by &"
      text: "&#nbsp;"
      expected: "&#nbsp;"

    - description: "Autolink hashtag followed by ! (! not included)"
      text: "text #hashtag!"
      expected: "text <a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">#hashtag</a>!"

    - description: "Autolink two hashtags separated by a slash"
      text: "text #dodge/#answer"
      expected: "text <a href=\"https://twitter.com/search?q=%23dodge\" title=\"#dodge\" class=\"tweet-url hashtag\">#dodge</a>/<a href=\"https://twitter.com/search?q=%23answer\" title=\"#answer\" class=\"tweet-url hashtag\">#answer</a>"

    - description: "Autolink hashtag before a slash"
      text: "text #dodge/answer"
      expected: "text <a href=\"https://twitter.com/search?q=%23dodge\" title=\"#dodge\" class=\"tweet-url hashtag\">#dodge</a>/answer"

    - description: "Autolink hashtag after a slash"
      text: "text dodge/#answer"
      expected: "text dodge/<a href=\"https://twitter.com/search?q=%23answer\" title=\"#answer\" class=\"tweet-url hashtag\">#answer</a>"

    - description: "Autolink hashtag followed by Japanese"
      text: "text #hashtagの"
      expected: "text <a href=\"https://twitter.com/search?q=%23hashtagの\" title=\"#hashtagの\" class=\"tweet-url hashtag\">#hashtagの</a>"

    - description: "Autolink hashtag preceded by full-width space (U+3000)"
      text: "text　#hashtag"
      expected: "text　<a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">#hashtag</a>"

    - description: "Autolink hashtag followed by full-width space (U+3000)"
      text: "#hashtag　text"
      expected: "<a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">#hashtag</a>　text"

    - description: "Autolink hashtag with full-width hash (U+FF03)"
      text: "＃hashtag"
      expected: "<a href=\"https://twitter.com/search?q=%23hashtag\" title=\"#hashtag\" class=\"tweet-url hashtag\">＃hashtag</a>"

    - description: "Autolink hashtag with accented character at the start"
      text: "#éhashtag"
      expected: "<a href=\"https://twitter.com/search?q=%23éhashtag\" title=\"#éhashtag\" class=\"tweet-url hashtag\">#éhashtag</a>"

    - description: "Autolink hashtag with accented character at the end"
      text: "#hashtagé"
      expected: "<a href=\"https://twitter.com/search?q=%23hashtagé\" title=\"#hashtagé\" class=\"tweet-url hashtag\">#hashtagé</a>"

    - description: "Autolink hashtag with accented character in the middle"
      text: "#hashétag"
      expected: "<a href=\"https://twitter.com/search?q=%23hashétag\" title=\"#hashétag\" class=\"tweet-url hashtag\">#hashétag</a>"

    - description: "Autolink hashtags in Korean"
      text: "What is #트위터 anyway?"
      expected: "What is <a href=\"https://twitter.com/search?q=%23트위터\" title=\"#트위터\" class=\"tweet-url hashtag\">#트위터</a> anyway?"

    - description: "Autolink hashtags in Russian"
      text: "What is #ашок anyway?"
      expected: "What is <a href=\"https://twitter.com/search?q=%23ашок\" title=\"#ашок\" class=\"tweet-url hashtag\">#ашок</a> anyway?"

    - description: "Autolink a katakana hashtag preceded by a space and followed by a space"
      text: "カタカナ #カタカナ カタカナ"
      expected: "カタカナ <a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a> カタカナ"

    - description: "Autolink a katakana hashtag preceded by a space and followed by a bracket"
      text: "カタカナ #カタカナ」カタカナ"
      expected: "カタカナ <a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>」カタカナ"

    - description: "Autolink a katakana hashtag preceded by a space and followed by a edge"
      text: "カタカナ #カタカナ"
      expected: "カタカナ <a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>"

    - description: "Autolink a katakana hashtag preceded by a bracket and followed by a space"
      text: "カタカナ「#カタカナ カタカナ"
      expected: "カタカナ「<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a> カタカナ"

    - description: "Autolink a katakana hashtag preceded by a bracket and followed by a bracket"
      text: "カタカナ「#カタカナ」カタカナ"
      expected: "カタカナ「<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>」カタカナ"

    - description: "Autolink a katakana hashtag preceded by a bracket and followed by a edge"
      text: "カタカナ「#カタカナ"
      expected: "カタカナ「<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>"

    - description: "Autolink a katakana hashtag preceded by a edge and followed by a space"
      text: "#カタカナ カタカナ"
      expected: "<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a> カタカナ"

    - description: "Autolink a katakana hashtag preceded by a edge and followed by a bracket"
      text: "#カタカナ」カタカナ"
      expected: "<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>」カタカナ"

    - description: "Autolink a katakana hashtag preceded by a edge and followed by a edge"
      text: "#カタカナ"
      expected: "<a href=\"https://twitter.com/search?q=%23カタカナ\" title=\"#カタカナ\" class=\"tweet-url hashtag\">#カタカナ</a>"

    - description: "Autolink a katakana hashtag with a voiced sounds mark followed by a space"
      text: "#ﾊｯｼｭﾀｸﾞ　テスト"
      expected: "<a href=\"https://twitter.com/search?q=%23ﾊｯｼｭﾀｸﾞ\" title=\"#ﾊｯｼｭﾀｸﾞ\" class=\"tweet-url hashtag\">#ﾊｯｼｭﾀｸﾞ</a>　テスト"

    - description: "Autolink a katakana hashtag with a voiced sounds mark followed by numbers"
      text: "#ﾊｯｼｭﾀｸﾞ123"
      expected: "<a href=\"https://twitter.com/search?q=%23ﾊｯｼｭﾀｸﾞ123\" title=\"#ﾊｯｼｭﾀｸﾞ123\" class=\"tweet-url hashtag\">#ﾊｯｼｭﾀｸﾞ123</a>"

    - description: "Autolink a katakana hashtag with another voiced sounds mark"
      text: "#ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ"
      expected: "<a href=\"https://twitter.com/search?q=%23ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ\" title=\"#ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ\" class=\"tweet-url hashtag\">#ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ</a>"

    - description: "Autolink a kanji hashtag preceded by a space and followed by a space"
      text: "漢字 #漢字 漢字"
      expected: "漢字 <a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a> 漢字"

    - description: "Autolink a kanji hashtag preceded by a space and followed by a bracket"
      text: "漢字 #漢字」漢字"
      expected: "漢字 <a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>」漢字"

    - description: "Autolink a kanji hashtag preceded by a space and followed by a edge"
      text: "漢字 #漢字"
      expected: "漢字 <a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>"

    - description: "Autolink a kanji hashtag preceded by a bracket and followed by a space"
      text: "漢字「#漢字 漢字"
      expected: "漢字「<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a> 漢字"

    - description: "Autolink a kanji hashtag preceded by a bracket and followed by a bracket"
      text: "漢字「#漢字」漢字"
      expected: "漢字「<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>」漢字"

    - description: "Autolink a kanji hashtag preceded by a bracket and followed by a edge"
      text: "漢字「#漢字"
      expected: "漢字「<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>"

    - description: "Autolink a kanji hashtag preceded by a edge and followed by a space"
      text: "#漢字 漢字"
      expected: "<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a> 漢字"

    - description: "Autolink a kanji hashtag preceded by a edge and followed by a bracket"
      text: "#漢字」漢字"
      expected: "<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>」漢字"

    - description: "Autolink a kanji hashtag preceded by a edge and followed by a edge"
      text: "#漢字"
      expected: "<a href=\"https://twitter.com/search?q=%23漢字\" title=\"#漢字\" class=\"tweet-url hashtag\">#漢字</a>"

    - description: "Autolink a kanji hashtag preceded by an ideographic comma, followed by an ideographic period"
      text: "これは、＃大丈夫。"
      expected: "これは、<a href=\"https://twitter.com/search?q=%23大丈夫\" title=\"#大丈夫\" class=\"tweet-url hashtag\">＃大丈夫</a>。"

    - description: "Autolink a hiragana hashtag preceded by a space and followed by a space"
      text: "ひらがな #ひらがな ひらがな"
      expected: "ひらがな <a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a> ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a space and followed by a bracket"
      text: "ひらがな #ひらがな」ひらがな"
      expected: "ひらがな <a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>」ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a space and followed by a edge"
      text: "ひらがな #ひらがな"
      expected: "ひらがな <a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>"

    - description: "Autolink a hiragana hashtag preceded by a bracket and followed by a space"
      text: "ひらがな「#ひらがな ひらがな"
      expected: "ひらがな「<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a> ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a bracket and followed by a bracket"
      text: "ひらがな「#ひらがな」ひらがな"
      expected: "ひらがな「<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>」ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a bracket and followed by a edge"
      text: "ひらがな「#ひらがな"
      expected: "ひらがな「<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>"

    - description: "Autolink a hiragana hashtag preceded by a edge and followed by a space"
      text: "#ひらがな ひらがな"
      expected: "<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a> ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a edge and followed by a bracket"
      text: "#ひらがな」ひらがな"
      expected: "<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>」ひらがな"

    - description: "Autolink a hiragana hashtag preceded by a edge and followed by a edge"
      text: "#ひらがな"
      expected: "<a href=\"https://twitter.com/search?q=%23ひらがな\" title=\"#ひらがな\" class=\"tweet-url hashtag\">#ひらがな</a>"

    - description: "Autolink a Kanji/Katakana mix hashtag"
      text: "日本語ハッシュタグ #日本語ハッシュタグ"
      expected: "日本語ハッシュタグ <a href=\"https://twitter.com/search?q=%23日本語ハッシュタグ\" title=\"#日本語ハッシュタグ\" class=\"tweet-url hashtag\">#日本語ハッシュタグ</a>"

    - description: "DO NOT autolink a hashtag without a preceding space"
      text: "日本語ハッシュタグ#日本語ハッシュタグ"
      expected: "日本語ハッシュタグ#日本語ハッシュタグ"

    - description: "DO NOT include a punctuation in a hashtag"
      text: "#日本語ハッシュタグ。"
      expected: "<a href=\"https://twitter.com/search?q=%23日本語ハッシュタグ\" title=\"#日本語ハッシュタグ\" class=\"tweet-url hashtag\">#日本語ハッシュタグ</a>。"

    - description: "Autolink a hashtag after a punctuation"
      text: "日本語ハッシュタグ。#日本語ハッシュタグ"
      expected: "日本語ハッシュタグ。<a href=\"https://twitter.com/search?q=%23日本語ハッシュタグ\" title=\"#日本語ハッシュタグ\" class=\"tweet-url hashtag\">#日本語ハッシュタグ</a>"

    - description: "Autolink a hashtag with chouon"
      text: "長音ハッシュタグ。#サッカー"
      expected: "長音ハッシュタグ。<a href=\"https://twitter.com/search?q=%23サッカー\" title=\"#サッカー\" class=\"tweet-url hashtag\">#サッカー</a>"

    - description: "Autolink a hashtag with half-width chouon"
      text: "長音ハッシュタグ。#ｻｯｶｰ"
      expected: "長音ハッシュタグ。<a href=\"https://twitter.com/search?q=%23ｻｯｶｰ\" title=\"#ｻｯｶｰ\" class=\"tweet-url hashtag\">#ｻｯｶｰ</a>"

    - description: "Autolink a hashtag with half-width # after full-width ！"
      text: "できましたよー！#日本語ハッシュタグ。"
      expected: "できましたよー！<a href=\"https://twitter.com/search?q=%23日本語ハッシュタグ\" title=\"#日本語ハッシュタグ\" class=\"tweet-url hashtag\">#日本語ハッシュタグ</a>。"

    - description: "Autolink a hashtag with full-width ＃ after full-width ！"
      text: "できましたよー！＃日本語ハッシュタグ。"
      expected: "できましたよー！<a href=\"https://twitter.com/search?q=%23日本語ハッシュタグ\" title=\"#日本語ハッシュタグ\" class=\"tweet-url hashtag\">＃日本語ハッシュタグ</a>。"

    - description: "Autolink a hashtag containing ideographic iteration mark"
      text: "#云々"
      expected: "<a href=\"https://twitter.com/search?q=%23云々\" title=\"#云々\" class=\"tweet-url hashtag\">#云々</a>"

    - description: "Autolink multiple hashtags in multiple languages"
      text: "Hashtags in #中文, #日本語, #한국말, and #русский! Try it out!"
      expected: "Hashtags in <a href=\"https://twitter.com/search?q=%23中文\" title=\"#中文\" class=\"tweet-url hashtag\">#中文</a>, <a href=\"https://twitter.com/search?q=%23日本語\" title=\"#日本語\" class=\"tweet-url hashtag\">#日本語</a>, <a href=\"https://twitter.com/search?q=%23한국말\" title=\"#한국말\" class=\"tweet-url hashtag\">#한국말</a>, and <a href=\"https://twitter.com/search?q=%23русский\" title=\"#русский\" class=\"tweet-url hashtag\">#русский</a>! Try it out!"

    - description: "Autolink should allow for ş (U+015F) in a hashtag"
      text: "Here’s a test tweet for you: #Ateş #qrşt #ştu #ş"
      expected: "Here’s a test tweet for you: <a href=\"https://twitter.com/search?q=%23Ateş\" title=\"#Ateş\" class=\"tweet-url hashtag\">#Ateş</a> <a href=\"https://twitter.com/search?q=%23qrşt\" title=\"#qrşt\" class=\"tweet-url hashtag\">#qrşt</a> <a href=\"https://twitter.com/search?q=%23ştu\" title=\"#ştu\" class=\"tweet-url hashtag\">#ştu</a> <a href=\"https://twitter.com/search?q=%23ş\" title=\"#ş\" class=\"tweet-url hashtag\">#ş</a>"

    - description: "Autolink a hashtag with Latin extended character"
      text: "#mûǁae"
      expected: "<a href=\"https://twitter.com/search?q=%23mûǁae\" title=\"#mûǁae\" class=\"tweet-url hashtag\">#mûǁae</a>"

# Please be careful with changes to this test case - what looks like "á" is really a + U+0301, and many editors will silently convert this to U+00E1.
    - description: "Autolink hashtags with combining diacritics"
      text: "#táim #hag̃ua"
      expected: "<a href=\"https://twitter.com/search?q=%23táim\" title=\"#táim\" class=\"tweet-url hashtag\">#táim</a> <a href=\"https://twitter.com/search?q=%23hag̃ua\" title=\"#hag̃ua\" class=\"tweet-url hashtag\">#hag̃ua</a>"

    - description: "Autolink Arabic hashtag"
      text: "Arabic hashtag: #فارسی #لس_آنجلس"
      expected: "Arabic hashtag: <a href=\"https://twitter.com/search?q=%23فارسی\" title=\"#فارسی\" class=\"tweet-url hashtag rtl\">#فارسی</a> <a href=\"https://twitter.com/search?q=%23لس_آنجلس\" title=\"#لس_آنجلس\" class=\"tweet-url hashtag rtl\">#لس_آنجلس</a>"

    - description: "Autolink Thai hashtag"
      text: "Thai hashtag: #รายละเอียด"
      expected: "Thai hashtag: <a href=\"https://twitter.com/search?q=%23รายละเอียด\" title=\"#รายละเอียด\" class=\"tweet-url hashtag\">#รายละเอียด</a>"
"""

import re

lines = hash_tag_examples.split("\n")
examples = []
for i, line in enumerate(lines):
    if line.startswith("    - description"):
        rex = re.compile(r'<a.*? title="(.*?)" .*?<\/a>')
        match = rex.findall(lines[i + 2][17:-1])
        examples.append(
            {
                "desc": lines[i],
                "text": lines[i + 1][13:-1],
                "expected": lines[i + 2][17:-1],
                "hashtags": match,
            },
        )


for i, d in enumerate(examples):
    print("i", i)
    hashtags = list(find_hashtags(d["text"]))
    for h in hashtags:
        if h not in d["hashtags"]:
            raise ValueError
    for h in d["hashtags"]:
        if h not in hashtags:
            raise Exception

d["desc"]
d["text"]
d["expected"]
hashtags
str.encode(d["text"][0]) in {str.encode("#")}
str.encode("#️")
str.encode("#hashtags")

"#️⃣"

list(find_hashtags("#️⃣yolo"))

for i in d["text"]:
    print(i, i.isalnum())
str.encode(d["text"][22])

str.encode("áim"[1])
"â"[0]


hashtag_tests = [
    ("text #hashtag", ["#hashtag"]),
    ("text #hash0tag", ["#hash0tag"]),
    ("text #1tag", ["#1tag"]),
    ("text #hash_tag", ["#hash_tag"]),
    ("text #1234", []),
    ("text#hashtag", []),
    ("#️hashtag", []),
    ("#⃣hashtag", []),
    ("text #hashtag1 #hashtag2", ["#hashtag1", "#hashtag2"]),
    ("text.#hashtag", ["#hashtag"]),
    ("&#nbsp;", []),
    ("text #hashtag!", ["#hashtag"]),
    ("text #dodge/#answer", ["#dodge", "#answer"]),
    ("text #dodge/answer", ["#dodge"]),
    ("text dodge/#answer", ["#answer"]),
    ("text #hashtagの", ["#hashtagの"]),
    ("text\u3000#hashtag", ["#hashtag"]),
    ("#hashtag\u3000text", ["#hashtag"]),
    ("＃hashtag", ["#hashtag"]),
    ("#éhashtag", ["#éhashtag"]),
    ("#hashtagé", ["#hashtagé"]),
    ("#hashétag", ["#hashétag"]),
    ("What is #트위터 anyway?", ["#트위터"]),
    ("What is #ашок anyway?", ["#ашок"]),
    ("カタカナ #カタカナ カタカナ", ["#カタカナ"]),
    ("カタカナ #カタカナ」カタカナ", ["#カタカナ"]),
    ("カタカナ #カタカナ", ["#カタカナ"]),
    ("カタカナ「#カタカナ カタカナ", ["#カタカナ"]),
    ("カタカナ「#カタカナ」カタカナ", ["#カタカナ"]),
    ("カタカナ「#カタカナ", ["#カタカナ"]),
    ("#カタカナ カタカナ", ["#カタカナ"]),
    ("#カタカナ」カタカナ", ["#カタカナ"]),
    ("#カタカナ", ["#カタカナ"]),
    ("#ﾊｯｼｭﾀｸﾞ\u3000テスト", ["#ﾊｯｼｭﾀｸﾞ"]),
    ("#ﾊｯｼｭﾀｸﾞ123", ["#ﾊｯｼｭﾀｸﾞ123"]),
    ("#ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ", ["#ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ"]),
    ("漢字 #漢字 漢字", ["#漢字"]),
    ("漢字 #漢字」漢字", ["#漢字"]),
    ("漢字 #漢字", ["#漢字"]),
    ("漢字「#漢字 漢字", ["#漢字"]),
    ("漢字「#漢字」漢字", ["#漢字"]),
    ("漢字「#漢字", ["#漢字"]),
    ("#漢字 漢字", ["#漢字"]),
    ("#漢字」漢字", ["#漢字"]),
    ("#漢字", ["#漢字"]),
    ("これは、＃大丈夫。", ["#大丈夫"]),
    ("ひらがな #ひらがな ひらがな", ["#ひらがな"]),
    ("ひらがな #ひらがな」ひらがな", ["#ひらがな"]),
    ("ひらがな #ひらがな", ["#ひらがな"]),
    ("ひらがな「#ひらがな ひらがな", ["#ひらがな"]),
    ("ひらがな「#ひらがな」ひらがな", ["#ひらがな"]),
    ("ひらがな「#ひらがな", ["#ひらがな"]),
    ("#ひらがな ひらがな", ["#ひらがな"]),
    ("#ひらがな」ひらがな", ["#ひらがな"]),
    ("#ひらがな", ["#ひらがな"]),
    ("日本語ハッシュタグ #日本語ハッシュタグ", ["#日本語ハッシュタグ"]),
    ("日本語ハッシュタグ#日本語ハッシュタグ", []),
    ("#日本語ハッシュタグ。", ["#日本語ハッシュタグ"]),
    ("日本語ハッシュタグ。#日本語ハッシュタグ", ["#日本語ハッシュタグ"]),
    ("長音ハッシュタグ。#サッカー", ["#サッカー"]),
    ("長音ハッシュタグ。#ｻｯｶｰ", ["#ｻｯｶｰ"]),
    ("できましたよー！#日本語ハッシュタグ。", ["#日本語ハッシュタグ"]),
    ("できましたよー！＃日本語ハッシュタグ。", ["#日本語ハッシュタグ"]),
    ("#云々", ["#云々"]),
    (
        "Hashtags in #中文, #日本語, #한국말, and #русский! Try it out!",
        ["#中文", "#日本語", "#한국말", "#русский"],
    ),
    (
        "Here’s a test tweet for you: #Ateş #qrşt #ştu #ş",
        ["#Ateş", "#qrşt", "#ştu", "#ş"],
    ),
    ("#mûǁae", ["#mûǁae"]),
    ("#táim #hag̃ua", ["#táim", "#hag̃ua"]),
    ("Arabic hashtag: #فارسی #لس_آنجلس", ["#فارسی", "#لس_آنجلس"]),
    ("Thai hashtag: #รายละเอียด", ["#รายละเอียด"]),
]
