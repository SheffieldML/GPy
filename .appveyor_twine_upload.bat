IF "%APPVEYOR_REPO_BRANCH%"=="deploy" (
  ECHO "twine upload --skip-existing dist/*" 1>&2
) ELSE (
  ECHO Only deploy on deploy branch
)



  echo not deploying on other branches, other than deploy



#ps: >-
#    If ($env:APPVEYOR_REPO_BRANCH -eq 'devel') { 
#        echo not deploying on devel # twine upload --skip-existing -r test dist/*
#    }
#    ElseIf ($env:APPVEYOR_REPO_BRANCH -eq 'deploy') {  
#        twine upload --skip-existing dist/*
#    }
#    Else {
#        echo not deploying on other branches
#    }