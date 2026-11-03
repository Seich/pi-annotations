# pi-annotations

Experimental memory extension. This extension is where I've been testing various memory experiments. The simplest approach is the best, to that end I am going with providing simple tools and guidelines to allow agents to make "mental notes".

The idea being that providing _JIT_ context to the agent allows for as little information as possible to be given to the agent while keeping all relevant context available.

After a bit of trial and error, here's my current approach:

- Provide a tool to make annotations: `annotate`, it allows creating, reading, updating and deleting notes on the code. This allows the agent to skip comments and still keep context around. We allow annotations to be anchored to files, symbols and lines. By providing basic guidelines it can keep as many annotations as it wants so it can remember things in the future e.g. _why we chose this particular db_.
  
  We store these as json files. This makes conflict easy to resolve via any VCS. We originally intended to use the git object but having to sync them separately lead of various issues.

- Enhance existing tools, like read and detect when bash tools like sed, grep or cat are used to provide back relevant annotations. When one of these is used we check what was read and provide the annotations for those particular files, lines and symbols that were returned.
- Results Gate, if we let the agent write any number of notes it’s possible for itself to provide too much irrelevant context for the task at hand when reading a file. This generally lowers the quality of work and increases expenses. 
  
  To combat this a gate is provided, the extensions allows for a local classifier model to be installed, such as [Laya](https://huggingface.co/convaiinnovations/laya) or the use of an api one such as Jev. This classifier model is provided the current task as context and asked to evaluate all possible annotations to determine if they are relevant to the current task. 

This extension is pretty minimal. Installing it on pi will provide the following commands:  

- `/annotations` to manage existing annotations on the current path.
- `/annotations-view` to see annotations inline with their respective anchor.
- `/annotations-export` to export all annotations and their respective code as an html file. 
- `/annotations-gate` to manage the classifier, it let’s you configure what model to use (api vs local) and to tweak the thresholds. It can also be disabled altogether. `/annotations-gate status` will show some helpful debug information to tweak the numbers if needed.

# Etc.
So far this extension has worked pretty well on my testing but it’s proven hard to quantify. I’ve tested it on my benchmark project and it allows for single tasks to be completed without much additional guidance it’ll be informed of things like style and related files without the need of a large agents.md file.