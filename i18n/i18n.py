import json
import locale
import os
from tools.file_io import read_text


def load_language_list(language):
    return json.loads(read_text(f"./i18n/locale/{language}.json"))


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)
        english_map = (
            self.language_map
            if language == "en_US"
            else load_language_list("en_US")
        )
        self.english_to_key = {value: key for key, value in english_map.items()}

    def __call__(self, key):
        if key in self.language_map:
            return self.language_map[key]
        original = self.english_to_key.get(key)
        if original is not None:
            return self.language_map.get(original, key)
        return key

    def __repr__(self):
        return "Use Language: " + self.language
