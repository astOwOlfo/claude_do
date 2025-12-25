from hashlib import sha256
import ast
import tokenize
import importlib.util
import importlib.metadata
from inspect import getsource
from os.path import isfile, join
from os import makedirs
from io import StringIO
import re
import json
from functools import partial
from dataclasses import dataclass
from typing import Any, Literal, Callable, Union, Optional

from claude_do.chat_completion import chat_completion


@dataclass(frozen=True, slots=True)
class ClaudeDoArgument:
    value: Any
    type: Union[str, Literal["not_provided"]] = "not_provided"
    description: Optional[str] = None

    def __post_init__(self) -> None:
        assert isinstance(self.type, str), "`ClaudeDoArgument.type` should be a string."


filename_to_imported_symbol: dict[str, Union[Callable, type]] = {}


PROMPT = """Please implement a function or class with the following signature.

- The function's or class's name should be exactly the same as in the signature below - do not change the name.
- You should write a self-contained python file with the function or class and all imports and auxiliary code needed for it. The file will be executed as is and the function will be imported from it.
    * Do not leave any things in the code that the user should modify, e.g. `your_base_url_here`.
    * Do not run the function or class on example inputs on the top level.
    * Put the file in a section foramtted exactly as follows:
<code>
Your code here
</code>
- Avoid using global state when possible.
- Every time you make ANY assumption, assert that the assumption is true. Write clear error messages if the assumption is false.
- If types are not provided for some arguments or the return type is not provided, add detailed type hints and always assert that the arguments indeed have those types.
    * They type hints should be detailed, e.g. `dict[str, list[int] | None]`, not just `dict`.
    * The type asserts should be comprehensive, e.g. for the example above, `assert isinstance(arg, dict) and all(isinstance(key, str) and (value is None or (isinstance(value, list) and all(isinstance(element, int) for element in value))) for key, value in arg.items())`, not just `assert isinstance(arg, dict)`.
- The following packages are already installed: {installed_packages}
    * Note that those are the names used with `pip install`, not the ones you should use with `import`, e.g. `scikit-learn` and not `sklearn` (but for most packages, the two names are the same).
    * All else equal, try to not use any packages that are not installed.
    * If using a package that is not installed is better in any way, add a section formatted as follows to your response:
<packages_to_install>
space separated list of packages
</packages_to_install>
    * Those names should be names used with `pip install`, not the ones used with `import`, e.g. `scikit-learn` and not `sklearn` (but for most packages, the two names are the same).
- If you ever want to make a change to your code, rewrite all the code in a new <code> ... </code> section.
    * Only the contents of the last <code> ... </code> section you write will be taken into account and all the other <code> ... </code> sections will be discarded.
    * This implies that every <code> ... </code> section you write should be self-contained and that you should not write extra <code> ... </code> sections with things like usage examples or alternative implementation suggestions.
- IMPORTANT: If anything is unclear, if you think the task is impossible or too hard for you, or if you think it is better not to do the task for any other reason, please include a section formatted exactly as follows in your response:
<dont>
Explanation of what the problem is.
</dont>
    * IMPORTANT: It is always better to use this feature than to write incorrect code.
    * IMPORTANT: You will not be penalized for using this feature.
    * If you find the instructions unclear, you can use this feature to ask for clarification, by writing "Please clarify blah" in the <dont> ... </dont> section. After that, you will be given a clarification and you will retry.
    * After you wrote your <code> ... </code>, please think again about whether you should write a <dont> ... </dont> section. You are allowed to write one even after you wrote a <code> ... </code> section.

The signature of the function or class you sould implement is:
```
{signature}
```
{additional_instructions_prompt}{context_prompt}
"""


def claude_do(
    prompt: str,
    *args,
    claude_do_return_type: Union[str, Literal["not_provided"]] = "not_provided",
    claude_do_generated_code_filename: Optional[str] = None,
    claude_do_model: str = "claude-opus-4-5",
    claude_do_max_tokens: int = 16384,
    claude_do_temperature: float = 1.0,
    claude_do_base_url: Optional[str] = None,
    claude_do_context_files: Optional[list[str]] = None,
    claude_do_context_definitions: Optional[list[Any]] = None,
    **kwargs,
) -> Any:
    for key in kwargs.keys():
        assert not key.startswith("claude_do_"), (
            f"`{key}` is not a valid argument to `claude_do`."
        )
    assert claude_do_return_type is not None, (
        '`claude_do_return_type` cannot be `None`. To not have a type hint, use `"not_provided"`. To have the type be None (as in the type hint `def f() -> None`), use the string `"None"`.'
    )

    decorated_args: list[ClaudeDoArgument] = [decorate_argument(arg) for arg in args]
    decorated_kwargs: dict[str, ClaudeDoArgument] = {
        key: decorate_argument(arg) for key, arg in kwargs.items()
    }
    bare_args: list[Any] = [arg.value for arg in decorated_args]
    bare_kwargs: dict[str, Any] = {
        key: arg.value for key, arg in decorated_kwargs.items()
    }

    if claude_do_generated_code_filename is None:
        hash: str = hash_claude_do_input(
            prompt=prompt,
            decorated_args=decorated_args,
            decorated_kwargs=decorated_kwargs,
            return_type=claude_do_return_type,
            model=claude_do_model,
            max_tokens=claude_do_max_tokens,
            temperature=claude_do_temperature,
            base_url=claude_do_base_url,
            context_files=claude_do_context_files,
            context_definitions=claude_do_context_definitions,
        )
        claude_do_generated_code_filename = f"{hash}.py"

    signature: str = function_signature(
        function_name="do_the_thing",
        docstring=prompt,
        decorated_args=decorated_args,
        decorated_kwargs=decorated_kwargs,
        return_type=claude_do_return_type,
    )

    function = claude_implement_signature(
        name="do_the_thing",
        signature=signature,
        additional_instructions=None,
        generated_code_filename=claude_do_generated_code_filename,
        model=claude_do_model,
        max_tokens=claude_do_max_tokens,
        temperature=claude_do_temperature,
        base_url=claude_do_base_url,
        context_files=claude_do_context_files,
        context_definitions=claude_do_context_definitions,
    )

    return function(*bare_args, **bare_kwargs)


def decorate_argument(arg: Any) -> ClaudeDoArgument:
    if isinstance(arg, ClaudeDoArgument):
        return arg
    return ClaudeDoArgument(value=arg)


def claude_implement(
    signature: Optional[Union[Callable, type]] = None,
    *,
    additional_instructions: Optional[str] = None,
    generated_code_filename: Optional[str] = None,
    model: str = "claude-opus-4-5",
    max_tokens: int = 16384,
    temperature: float = 1.0,
    base_url: Optional[str] = None,
    context_files: Optional[list[str]] = None,
    context_definitions: Optional[list[Any]] = None,
) -> Union[Callable, type]:
    def claude_implement_decorator(
        signature: Union[Callable, type], _generated_code_filename: Optional[str]
    ) -> Union[Callable, type]:
        name: str = signature.__name__
        signature_source_code: str = getsource(signature)
        signature_source_code = remove_claude_do_decorator(signature_source_code)

        if _generated_code_filename is None:
            hash: str = hash_claude_implement_input(
                name=name,
                signature=signature_source_code,
                additional_instructions=additional_instructions,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=base_url,
                context_files=context_files,
                context_definitions=context_definitions,
            )
            _generated_code_filename = f"{hash}.py"

        return claude_implement_signature(
            name=name,
            signature=signature_source_code,
            additional_instructions=additional_instructions,
            generated_code_filename=_generated_code_filename,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=base_url,
            context_files=context_files,
            context_definitions=context_definitions,
        )

    if signature is not None:
        return claude_implement_decorator(
            signature, _generated_code_filename=generated_code_filename
        )
    else:
        return partial(
            claude_implement_decorator, _generated_code_filename=generated_code_filename
        )


def has_hash_comments(code: str) -> bool:
    tokens = tokenize.generate_tokens(StringIO(code).readline)
    return any(t.type == tokenize.COMMENT for t in tokens)


def remove_claude_do_decorator(definition_source_code: str) -> str:
    assert not has_hash_comments(definition_source_code), (
        'Definitions decorated with `@claude_implement` should not have `# blah` comments, only `""" blah """` comments.'
    )

    error_message = "The AST of the definition decorated with `@claude_implement` has an unexpected structure. You probably decorated something other than a function definition or class definition with `@claude_implement`, or decorated a weird function or class definition."
    tree = ast.parse(definition_source_code)
    assert len(tree.body) == 1, error_message
    assert hasattr(tree.body[0], "decorator_list"), error_message
    decorator_list = tree.body[0].decorator_list  # type: ignore
    assert len(decorator_list) > 0, error_message

    def is_claude_implement_decorator(decorator) -> bool:
        # @claude_implement (without parentheses)
        if isinstance(decorator, ast.Name):
            return decorator.id == "claude_implement"
        # @claude_implement(...) (with paretheses)
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id == "claude_implement"
        return False

    assert any(
        is_claude_implement_decorator(decorator) for decorator in decorator_list
    ), error_message
    tree.body[0].decorator_list = [  # type: ignore
        decorator
        for decorator in decorator_list
        if not is_claude_implement_decorator(decorator)
    ]
    return ast.unparse(tree)


def claude_implement_signature(
    name: str,
    signature: str,
    additional_instructions,
    generated_code_filename: str,
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    context_files: Optional[list[str]],
    context_definitions: Optional[list[Any]],
) -> Union[Callable, type]:
    global filename_to_imported_symbol

    if context_definitions is not None:
        assert all(hasattr(symbol, "__name__") for symbol in context_definitions), (
            "All of the elements of `claude_do_context_definitions` should have a `__name__` attribute."
        )

    already_loaded: Optional[Union[Callable, type]] = filename_to_imported_symbol.get(
        generated_code_filename
    )
    if already_loaded is not None:
        return already_loaded

    dir = "claude_do_generated_code"
    code_filename: str = join(dir, generated_code_filename)

    first_time: bool = not isfile(code_filename)
    if first_time:
        complete_prompt = PROMPT.format(
            signature=signature,
            installed_packages=installed_packages_prompt(),
            additional_instructions_prompt="\n\nAdditional instructions\n"
            + additional_instructions
            if additional_instructions is not None
            else "",
            context_prompt=context_files_prompt(context_files)
            + context_definitions_prompt(context_definitions),
        )

        print("\033[1;33mClaude Do: Waiting for Claude API call.\033[0m")
        response: str = chat_completion(
            complete_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=base_url,
        )

        parsed_response: ParsedResponse = parse_claude_response(response)

        if len(parsed_response.donts) > 0:
            print_donts_and_fail(parsed_response.donts)
        assert parsed_response.code is not None

        assert not isfile(code_filename), (
            "A claude_do file was created while generating a completion with Claude, most probably because you called claude_do multiple times in parallel, which you should not do. "
        )

        makedirs(dir, exist_ok=True)
        with open(code_filename, "w") as f:
            f.write(parsed_response.code)

    if first_time and parsed_response.packages_to_install is not None:  # type: ignore
        print_please_install_packages_and_fail(parsed_response.packages_to_install)  # type: ignore

    implemented: Union[Callable, type] = import_from_file(
        filename=code_filename,
        symbol_name=name,
        namespace_definitions=context_definitions,
    )
    filename_to_imported_symbol[generated_code_filename] = implemented

    return implemented


def import_from_file(
    filename: str, symbol_name: str, namespace_definitions: Optional[list[Any]]
) -> Union[Callable, type]:
    error_message = f"claude_do internal error: failed load python file `{filename}`."
    spec = importlib.util.spec_from_file_location("module", filename)
    assert spec is not None, error_message
    assert spec.loader is not None, error_message
    module = importlib.util.module_from_spec(spec)
    if namespace_definitions is not None:
        for symbol in namespace_definitions:
            setattr(module, symbol.__name__, symbol)
    spec.loader.exec_module(module)
    try:
        function = getattr(module, symbol_name)
    except AttributeError:
        assert False, (
            f"claude_do internal error: failed import symbol `{symbol_name}` from file `{filename}`."
        )
    return function


def hash_claude_do_input(
    prompt: str,
    decorated_args: list[ClaudeDoArgument],
    decorated_kwargs: dict[str, ClaudeDoArgument],
    return_type: Union[str, Literal["not_provided"]],
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    context_files: Optional[list[str]],
    context_definitions: Optional[list[Any]],
) -> str:
    json_input = {
        "prompt": prompt,
        "args": [[arg.type, arg.description] for arg in decorated_args],
        "kwargs": sorted(
            [
                [key, str(arg.type), arg.description]
                for key, arg in decorated_kwargs.items()
            ],
            key=lambda key_type_desc: key_type_desc[0],  # type: ignore
        ),
        "return_type": return_type,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "base_url": base_url,
        "context_files": context_files,
        "context_definition_names": [
            getattr(symbol, "__name__", None) for symbol in context_definitions
        ]
        if context_definitions is not None
        else None,
    }

    return sha256(json.dumps(json_input).encode()).hexdigest()


def hash_claude_implement_input(
    name: str,
    signature: str,
    additional_instructions: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    context_files: Optional[list[str]],
    context_definitions: Optional[list[Any]],
) -> str:
    json_input = {
        "name": name,
        "signature": signature,
        "additional_instructions": additional_instructions,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "base_url": base_url,
        "context_files": context_files,
        "context_definition_names": [
            getattr(symbol, "__name__", None) for symbol in context_definitions
        ]
        if context_definitions is not None
        else None,
    }

    return sha256(json.dumps(json_input).encode()).hexdigest()


def function_signature(
    function_name: str,
    docstring: str,
    decorated_args: list[ClaudeDoArgument],
    decorated_kwargs: dict[str, ClaudeDoArgument],
    return_type: Union[str, Literal["not_provided"]],
) -> str:
    names_and_decorated_args: list[tuple[str, ClaudeDoArgument]] = [
        (f"positional_argument_{i}", arg) for i, arg in enumerate(decorated_args)
    ] + list(decorated_kwargs.items())

    signature = f"def {function_name}("

    str_args: list[str] = [
        name + (f": {arg.type}" if arg.type != "not_provided" else "")
        for name, arg in names_and_decorated_args
    ]
    signature += ", ".join(str_args)
    signature += ")"

    if return_type != "not_provided":
        signature += f" -> {return_type}"
    signature += ":\n"

    signature += '"""\n'
    signature += docstring

    signature += "\n\nArguments:\n"
    for name, arg in names_and_decorated_args:
        signature += f"    {name}"
        if arg.type != "not_provided":
            signature += f": {arg.type}"
        signature += "\n"
        if arg.description is not None:
            signature += f"    {arg.description}\n"
        signature += f"    Example: {truncate(str(arg.value), 256)}\n\n"
    signature = signature.removesuffix("\n")

    signature += '"""\n'

    return signature


def truncate(s: str, max_characters: int) -> str:
    if len(s) <= max_characters:
        return s
    return s[: max_characters // 2] + "..." + s[-max_characters // 2 :]


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    code: Optional[str]
    donts: list[str]
    packages_to_install: Optional[str]


def parse_claude_response(response: str) -> ParsedResponse:
    codes: list[str] = extract_xml_sections(response, tag_name="code")
    donts: list[str] = extract_xml_sections(response, tag_name="dont")
    packages_to_install_lists: list[str] = extract_xml_sections(
        response, tag_name="packages_to_install"
    )

    assert not (len(codes) == 0 and len(donts) == 0), (
        "Claude Do: Claude gave an invalid response: it did not generate any <code> ... </code> or <dont> ... </dont> sections."
    )

    code = codes[-1] if len(codes) > 0 else None

    packages_to_install: list[str] = list(
        set(
            package
            for package_list in packages_to_install_lists
            for package in package_list.split()
        )
    )

    return ParsedResponse(
        code=code,
        donts=donts,
        packages_to_install=" ".join(packages_to_install)
        if len(packages_to_install) > 0
        else None,
    )


def extract_xml_sections(s: str, tag_name: str) -> list[str]:
    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    matches = re.findall(pattern, s, re.DOTALL)
    return matches


def installed_packages() -> list[str]:
    return [
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    ]


def installed_packages_prompt() -> str:
    packages: list[str] = installed_packages()
    if len(packages) == 0:
        return "No packages are installed."
    return ", ".join(packages)


def print_donts_and_fail(donts: list[str]) -> None:
    print()
    for dont in donts:
        print(
            "\033[1;31mClaude Do: Claude did not generate a function and gave the following reason for why:\033[0m",
            dont,
        )
        print()

    assert False, (
        "Claude did not generate a function and told you why. You can see what Claude said printed above."
    )


def print_please_install_packages_and_fail(packages: str) -> None:
    print()
    print(
        "\033[1;33mClaude Do: Please install the following packages and rerun your code:\033[0m\033[33m",
        packages,
        "\033[0m",
    )
    print()

    assert False, (
        f"Claude Do: Please install packages '{packages}' and rerun your code."
    )


CONTEXT_DEFINITIONS_PROMPT = """

Here are some relevant functions and classes that you can use.
IMPORTANT: DO NOT rewrite those definitions in the code you generate or import them.
You should assume they are already part of the namespace the code is executed in.
This is because those definitions will be added automatically to your code when it is executed.
This is an exception to the "write self contained code" instruction above.

"""


def context_definitions_prompt(symbols: Optional[list[Any]]) -> str:
    if symbols is None or len(symbols) == 0:
        return ""

    return CONTEXT_DEFINITIONS_PROMPT + "\n\n".join(
        f"```python\n{getsource(symbol)}\n```" for symbol in symbols
    )


def context_files_prompt(files: Optional[list[str]]) -> str:
    if files is None or len(files) == 0:
        return ""

    contents: list[str] = []
    for file in files:
        assert isfile(file), (
            f"Claude Do: The file '{file}' that you gave in `claude_do_context_files` does not exist or is a directory."
        )
        with open(file) as f:
            contents.append(f.read())

    return (
        "\n\nHere are the contents of some files in the project that are relevant for the task:\n\n"
        + "\n\n".join(
            f"File `{file}` contains:\n```\n{content}\n```"
            for file, content in zip(files, contents, strict=True)
        )
    )
