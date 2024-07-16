# -*- coding: utf-8 -*-
"""
Types from C#.
"""
import sys, os, asyncio
from typing import Awaitable, Any
import clr

import System

global IListOf
IListOf = getattr(System.Collections.Generic, "IList`1")

global ActionOf
ActionOf = getattr(System, "Action`1")

global TaskOf
TaskOf = getattr(System.Threading.Tasks, "Task`1")

# use NoCancellation instead of Cancellation.None, because None is a keyword
global NoCancellation
NoCancellation = getattr(System.Threading.CancellationToken, 'None')


def _onLogMessage(line: str):
    sys.stdout.write(f'>>> {line}' + os.linesep)
    sys.stdout.flush()

_context = None


def __getattr__(name):
    if name == "context":
        global _context
        return _context

    raise AttributeError(f"module {__name__} has no attribute {name}")


def connect(host: str = None, fingerprint: str = None, path_to_dll: str = "") -> Awaitable[Any]:
    try:
        clr.AddReference(os.path.join(os.path.abspath(path_to_dll), "Nea.Client.PythonSDK.dll"))

    except System.IO.FileNotFoundException as e:
        sys.stderr.write(f'SDK files not found in current working directory {os.getcwd()}: {e.Message}\n')
        sys.exit(1)
    except Exception as e:
        sys.stderr.write('Failed to load the SDK: %s\n' % (e.Message))
        sys.exit(1)

    # 1. Cannot subclass from LabInfosAction because it is a delegate
    # 2. Async/Await/Future does not work (Deadlock)

    async def getContext(database_address, controller_id):
        try:
            import Nea.Client.PythonSDK as nea
            print("Creating the context")
            global _context
            _context = nea.PythonSDK.CreateApplicationContext(
                database_address, controller_id,
                ActionOf[str](lambda msg: _onLogMessage(msg)))
            return _context;
        except Exception as e:
            sys.stderr.write('Failed to get context: %s\n' % (e.Message))
            return None
        except:
            sys.stderr.write('Failed to get context.')
            return None

    task = asyncio.create_task(
        getContext(host, fingerprint))

    return task


def Handler(TaskResult):
    """
    Build an event handler.

    Parameters
    ----------
    TaskResult : TYPE
        Type of the result the task will return on completion.

    Returns
    -------
    Action<Task<T>>
        DESCRIPTION.

    """
    return ActionOf[TaskOf[TaskResult]]
