#Navier-Stokes

## Upload local folder:

Assume that the output of the command **git remote -v** is

```bash
591-esenonfossiio@rvpn-01-023:navier-stokes$ git remote -v 
luca	git@gitlab.com:Collaborators/navier-stokes.git (fetch)
luca	git@gitlab.com:Collaborators/navier-stokes.git (push)
origin	git@gitlab.com:ESeNonFossiIo/navier-stokes.git (fetch)
origin	git@gitlab.com:ESeNonFossiIo/navier-stokes.git (push)
```

in order to update your folder:

```
	git fetch --all
    git rebase luca <your-branch>
```

and then if you are not lucky, solve your conficts!
Once you have done it
```
	git rebase --continue
```

## Upload remote folder:

Once you have updated your local repo, you have to upload your changes on the remote repo. In order to do that you have to force it:

```
	git push -f origin <your-branch>
```

## Ulisse:

to load **deal.ii** and **deal.ii SAK** :

	. /home/mathlab/gnu.conf /home/mathlab/gnu/

## Indentation:

when you have made modifications to the code, make sure you run

```
	./script/indent
```
before committing, so that the indentation of the source code is consistent for all developers, and you don't commit space changes...

For this to work, you have to install astyle 2.04 (**exactly this version**). 
