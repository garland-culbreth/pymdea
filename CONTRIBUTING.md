# Contributing

Thanks for contributing! All contributions are welcome, from reporting bugs to implementing new features.

## Reporting bugs

Use [GitHub issues](https://github.com/garland-culbreth/network-infodemic-model/issues) to track bugs. You can report new bugs by creating a [new issue](https://github.com/garland-culbreth/network-infodemic-model/issues/new/choose).

Before creating a new bug report, check that the bug has not already been reported in an existing issue. If a closed issue seems related to the bug you're experiencing, create a new issue and link the closed one in the your issue description.

## Suggesting enhancements

Use [GitHub issues](https://github.com/garland-culbreth/network-infodemic-model/issues) to suggest enhancements. You can suggest new features by creating a [new issue](https://github.com/garland-culbreth/network-infodemic-model/issues/new/choose).

Before creating a new enhancement suggestion, check that the enhancement has not already been suggested in an existing issue.

## Contributing to the codebase

### Python style

All pull requests must pass a [ruff](https://docs.astral.sh/ruff/linter/) GitHub actions [workflow](https://github.com/garland-culbreth/network-echos/actions/workflows/ruff.yml) which enforces consistent style. Installing a ruff plugin for your IDE is recommended to check changes before pushing.

### Docstrings

All new code should be be documented. Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

### Tests

All new code should have tests. Tests for a module should cover everything in the module. All pull requests must pass a [pytest](https://docs.pytest.org/en/stable/) GitHub actions [workflow](https://github.com/garland-culbreth/network-echos/actions/workflows/pytest.yml), which runs all tests, before they can be merged. Before pushing changes, run all tests with pytest locally and confirm they pass.

## Version control

Use [git](https://git-scm.com/) for version control.

### Flow

Use [short-lived feature branches](https://trunkbaseddevelopment.com/short-lived-feature-branches/) for [trunk-based development](https://trunkbaseddevelopment.com/). Every branch should correspond to one change, or one collection of closely related changes, and should be merged to main within a few days.

### Branches

Branch names should be short and consisely describe the gist of changes being made. They should be prefixed with one of the [standard acronyms](#standard-acronyms) in lower case, followed by a forward slash `/`. Use hyphens `-` to separate words.

#### Standard acronyms

- `BLD`: Build. For changes related to builds.
- `DOC`: Documentation. For changes related to documentation.
- `ENH`: Enhancement. For new features or other chances that enhance functionality.
- `FIX`: Bug fix. For changes that fix bugs and broken functionality.
- `MNT`: Maintenance. For changes related to maintaining the codebase which don't add new functionality or fix broken functionality.
- `REL`: Release. For changes related to releases.
- `TST`: Test. For new tests or changes to existing tests.

#### Example branch names

```txt
enh/add-a-feature
fix/fix-a-bug
doc/write-new-docs
```

### Pull requests

Follow the pull request template.

Titles should be shorter than 80 characters, including spaces, and clearly state the purpose of the changes proposed. They should begin with a [standard icon](#standard-icons) and one of the [standard acronyms](#standard-acronyms) in all caps, followed by a hyphen ` - ` padded by one space on either side.

#### Standard icons

Emoji icons corresponding to the [standard acronyms](#standard-acronyms). Use these as the beginning of the prefix in pull request titles.

- `BLD`: 📦
- `DOC`: 📝
- `ENH`: 💡
- `FIX`: 🧯
- `MNT`: 🛠️
- `REL`: 🚀
- `TST`: 🧪

#### Example pull request titles

```txt
🧯FIX - Fix bug in some_function when some_parameter has forbidden value
💡ENH - Support some new useful thing
📝DOC - Correct some mis-spelled word on a page
🛠️MNT - Change formatter rule
```

### Merging

#### From a feature branch into the main branch

Use a merge commit when merging a feature branch into main.

#### From the main branch into a feature branch

Use rebase when merging new changes from main into an existing feature branch. This should rarely be needed.

## Contributing to the documentation

### Documentation style

Follow the [Google developer documentation style guide](https://developers.google.com/style/highlights).
