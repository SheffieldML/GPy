IF "%APPVEYOR_REPO_BRANCH%"=="deploy" (
  twine upload --skip-existing dist/*
) ELSE (
  ECHO Only deploy on deploy branch
)