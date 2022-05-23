# Docs development

When changing something in the `puma` documentation, you might find yourself in a
situation where you want to see if your changes have the intended effect.

The docs are only deployed for commits on the `main` branch. However, the docs are
built for _every_ commit, no matter on which branch, and are uploaded as an artifact.

This means that you can download the docs as a `.zip` file and then browser the html
files on your machine.

If you have an open pull request for you changes, you find the artifact like shown
below (click on the button that is marked with the red circle):

![](../assets/artifact_steps_1.png)
![](../assets/artifact_steps_2.png)
![](../assets/artifact_steps_3.png)
