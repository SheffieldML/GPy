IF "%APPVEYOR_REPO_BRANCH%"=="deploy" (
  ECHO "twine upload --skip-existing dist/*" 1>&2
) ELSE (
  ECHO Only deploy on deploy branch 1>&2
)