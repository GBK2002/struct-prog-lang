from tokenizer import tokenize

"""
parser.py -- implement parser for simple expressions

Accept a string of tokens, return an AST expressed as stack of dictionaries
"""

ebnf = """
    factor = <number> | "(" expression ")"
    term = factor { "*"|"/" factor }
    expression = term { "+"|"-" term }
    statement = <print> expression | expression
    program = expression
"""

bnf = """
    factor = <number>
    factor = <identifier>
    factor = "(" expression ")"

    term = factor { "*"|"/" factor }
    term = term * factor
    term = term / factor

    expression = term 
    expression = expression + term
    expression = expression - term

    statement = <print> expression
    statement = expression
"""

def parse_factor(tokens):
    """
    factor = <number> | "(" expression ")"
    """
    token = tokens[0]
    if token["tag"] == "number":
        return {
            "tag":"number",
            "value": token["value"]
        }, tokens[1:]
    if token["tag"] == "(":
        ast, tokens = parse_expression(tokens[1:])
        assert tokens[0]["tag"] == ")"
        return ast, tokens[1:]
    raise Exception(f"Unexpected token '{token['tag']}' at position {token['position']}.")

def test_parse_factor():
    """
    factor = <number> | "(" expression ")"
    """
    print("testing parse_factor()")
    for s in ["1","22","333"]:
        tokens = tokenize(s)
        ast, tokens = parse_factor(tokens)
        assert ast=={'tag': 'number', 'value': int(s)}
        assert tokens[0]['tag'] == None 
    for s in ["(1)","(22)"]:
        tokens = tokenize(s)
        ast, tokens = parse_factor(tokens)
        s_n = s.replace("(","").replace(")","")
        assert ast=={'tag': 'number', 'value': int(s_n)}
        assert tokens[0]['tag'] == None 
    tokens = tokenize("(2+3)")
    ast, tokens = parse_factor(tokens)
    assert ast == {'tag': '+', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 3}}

def parse_term(tokens):
    """
    term = factor { "*"|"/" factor }
    """
    node, tokens = parse_factor(tokens)
    while tokens[0]["tag"] in ["*","/"]:
        tag = tokens[0]["tag"]
        right_node, tokens = parse_factor(tokens[1:])
        node = {"tag":tag, "left":node, "right":right_node}

    return node, tokens

def test_parse_term():
    """
    term = factor { "*"|"/" factor }
    """
    print("testing parse_term()")
    for s in ["1","22","333"]:
        tokens = tokenize(s)
        ast, tokens = parse_term(tokens)
        assert ast=={'tag': 'number', 'value': int(s)}
        assert tokens[0]['tag'] == None 
    tokens = tokenize("2*4")
    ast, tokens = parse_term(tokens)
    assert ast == {'tag': '*', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 4}}
    tokens = tokenize("2*4/6")
    ast, tokens = parse_term(tokens)
    assert ast == {'tag': '/', 'left': {'tag': '*', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 4}}, 'right': {'tag': 'number', 'value': 6}}

def parse_expression(tokens):
    """
    expression = term { "+"|"-" term }
    """
    node, tokens = parse_term(tokens)
    while tokens[0]["tag"] in ["+","-"]:
        tag = tokens[0]["tag"]
        right_node, tokens = parse_term(tokens[1:])
        node = {"tag":tag, "left":node, "right":right_node}

    return node, tokens

def test_parse_expression():
    """
    expression = term { "+"|"-" term }
    """
    print("testing parse_expression()")
    for s in ["1","22","333"]:
        tokens = tokenize(s)
        ast, tokens = parse_expression(tokens)
        assert ast=={'tag': 'number', 'value': int(s)}
        assert tokens[0]['tag'] == None 
    tokens = tokenize("2*4")
    ast, tokens = parse_expression(tokens)
    assert ast == {'tag': '*', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 4}}
    tokens = tokenize("1+2*4")
    ast, tokens = parse_expression(tokens)
    assert ast == {'tag': '+', 'left': {'tag': 'number', 'value': 1}, 'right': {'tag': '*', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 4}}}
    tokens = tokenize("1+(2+3)*4")
    ast, tokens = parse_expression(tokens)
    assert ast == {'tag': '+', 'left': {'tag': 'number', 'value': 1}, 'right': {'tag': '*', 'left': {'tag': '+', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 3}}, 'right': {'tag': 'number', 'value': 4}}}

def parse_statement(tokens):
    """
    statement = <print> expression | expression
    """
    if tokens[0]["tag"] == "print":
        value_ast, tokens = parse_expression(tokens[1:])
        ast = {
            'tag':'print',
            'value': value_ast
        }

    else:
        ast, tokens = parse_expression(tokens)
    return ast, tokens

def test_parse_statement():
    """
    statement = <print> expression | expression
    """
    print("testing parse_statement()")
    tokens = tokenize("1+(2+3)*4")
    ast, tokens = parse_statement(tokens)
    assert ast == {'tag': '+', 'left': {'tag': 'number', 'value': 1}, 'right': {'tag': '*', 'left': {'tag': '+', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 3}}, 'right': {'tag': 'number', 'value': 4}}}
    tokens = tokenize("print 2*4")
    ast, tokens = parse_statement(tokens)
    assert ast == {'tag': 'print', 'value': {'tag': '*', 'left': {'tag': 'number', 'value': 2}, 'right': {'tag': 'number', 'value': 4}}}



def parse(tokens):
    """
        program = expression
    """
    ast, _ = parse_statement(tokens)
    return ast

def test_parse():
    """
        program = expression
    """
    print("testing parse()")
    tokens = tokenize("1+(2+3)*4")
    ast1, _ = parse_statement(tokens)
    ast2 = parse(tokens)
    assert ast1 == ast2, "parse() is not evaluating via parse_expression()"
def test_parse_factor_additional():
    """
    Additional test for parse_factor: nested parentheses
    """
    tokens = tokenize("((3))")
    ast, tokens = parse_factor(tokens)
    assert ast == {'tag': 'number', 'value': 3}
    assert tokens[0]['tag'] == None


def test_parse_term_additional():
    """
    Additional test for parse_term: chained division
    """
    tokens = tokenize("12/3/2")
    ast, tokens = parse_term(tokens)
    # left-associative: (12/3)/2
    assert ast == {
        'tag': '/',
        'left': {
            'tag': '/',
            'left': {'tag': 'number', 'value': 12},
            'right': {'tag': 'number', 'value': 3}
        },
        'right': {'tag': 'number', 'value': 2}
    }
    assert tokens[0]['tag'] == None




if __name__ == "__main__":
    test_parse_factor()
    test_parse_term()
    test_parse_expression()
    test_parse_statement()
    test_parse()
    test_parse_factor_additional()
    test_parse_term_additional()

    print("done.")
