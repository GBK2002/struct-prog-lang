import re

patterns = [
    [r"//[^\n]*", "comment"],  # Comment
    [r"\s+", "whitespace"],  # Whitespace
    [r"\d*\.\d+|\d+\.\d*|\d+", "number"],  # numeric literals
    [r'"([^"]|"")*"', "string"],  # string literals
    [r"true|false", "boolean"],  # boolean literals
    [r"null", "null"],  # the null literal
    [r"function", "function"],  # function keyword
    [r"return", "return"],  # return keyword
    [r"extern", "extern"],  # extern keyword
    [r"if", "if"],  # if keyword
    [r"else", "else"],  # else keyword
    [r"while", "while"],  # while keyword
    [r"for", "for"],  # for keyword
    [r"break", "break"],  # for keyword
    [r"continue", "continue"],  # for keyword
    [r"print", "print"],  # print keyword
    [r"import", "import"],  # import keyword
    [r"exit", "exit"],  # exit keyword
    [r"and", "&&"],  # alternate for &&
    [r"or", "||"],  # alternate for ||
    [r"not", "!"],  # alternate for !
    [r"assert", "assert"],
    [r"[a-zA-Z_][a-zA-Z0-9_]*", "identifier"],  # identifiers
    [r"\+", "+"],
    [r"\-", "-"],
    [r"\*", "*"],
    [r"\/", "/"],
    [r"\%", "%"],
    [r"\(", "("],
    [r"\)", ")"],
    [r"\{", "{"],
    [r"\}", "}"],
    [r"==", "=="],
    [r"!=", "!="],
    [r"<=", "<="],
    [r">=", ">="],
    [r"<", "<"],
    [r">", ">"],
    [r"\&\&", "&&"],
    [r"\|\|", "||"],
    [r"\!", "!"],
    [r"\=", "="],
    [r"\.", "."],
    [r"\[", "["],
    [r"\]", "]"],
    [r"\,", ","],
    [r"\:", ":"],
    [r"\;", ";"],
    [r".", "error"],  # unexpected content
]

for pattern in patterns:
    pattern[0] = re.compile(pattern[0])

test_generated_tags = set()

def tokenize(characters, generated_tags=test_generated_tags):
    line = 1
    tokens = []
    position = 0
    while position < len(characters):
        for pattern, tag in patterns:
            match = pattern.match(characters, position)
            if match:
                break
        assert match
        generated_tags.add(tag)
        if tag == "error":
            raise Exception(f"Syntax error: illegal character : {[match.group(0)]}")
        token = {"tag": tag, "position": position}
        value = match.group(0)
        if token["tag"] == "identifier":
            token["value"] = value
        if token["tag"] == "string":
            token["value"] = value[1:-1].replace('""', '"')
        if token["tag"] == "number":
            if "." in value:
                token["value"] = float(value)
            else:
                token["value"] = int(value)
        if token["tag"] == "boolean":
            token["value"] = True if value == "true" else False
        if tag == "whitespace":
            for c in value:
                if c == "\n":
                    line = line + 1
        token["line"] = line
        if tag not in ["comment", "whitespace"]:
            tokens.append(token)
        position = match.end()
    tokens.append({"tag": None, "position": position, "line": line})
    return tokens

if __name__ == "__main__":
    print("testing tokenizer.")
    print("done.")