#!/usr/bin/env python
import sys
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def main():
    environment = {}
    watch_var = None  # Variable to watch
    filename = None
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("watch="):
            watch_var = arg[6:]  # Extract identifier after "watch="
        else:
            filename = arg
    
    if filename:
        # Filename provided, read and execute it
        with open(filename, 'r') as f:
            source_code = f.read()
        try:
            tokens = tokenize(source_code)
            ast = parse(tokens)
            final_value, exit_status = evaluate(ast, environment, watch_var)
            if exit_status == "exit":
                sys.exit(final_value if isinstance(final_value, int) else 0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # REPL loop
        while True:
            try:
                source_code = input('>> ')
                if source_code.strip() in ['exit', 'quit']:
                    break
                tokens = tokenize(source_code)
                ast = parse(tokens)
                final_value, exit_status = evaluate(ast, environment, watch_var)
                if exit_status == "exit":
                    print(f"Exiting with code: {final_value}")
                    sys.exit(final_value if isinstance(final_value, int) else 0)
                elif final_value is not None:
                    print(final_value)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()