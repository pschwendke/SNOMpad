"""
Please edit configure.py before you run this example.

If you cannot access "context" in the Spyder Console.

    Run > Configuration per file

    [X] Run in console's namespace instead of an empty one
"""

import asyncio
import nest_asyncio
import neaspec


@asyncio.coroutine
async def main():
    """Run the demo."""

    import configuration as config

    # Use the controller and its database as automatically discovered in the network
    task = neaspec.connect(path_to_dll = config.dll)

    # Use the database (specified by the connection string)
    # and the controller (specified by the fingerprint GUID)
    # 
    # task = neaspec.connect(config.db, config.vm, config.dll)

    print("Waiting for context...");

    context = await task

    print("Ready. " +
          "Active Controller: " + context.Database.ActiveController.Name)

    # Option 1: ContinueWith callback handler ---------------------------------
    #
    # import Nea.Client.SharedDefinitions as neas
    #
    # context.Database.SelectLabs().ContinueWith(
    #    sdk.Handler(sdk.IListOf[neas.LabInfo])(
    #        lambda t: print(t.Result)))

    # Option 2: asyncio task --------------------------------------------------
    async def getLabs(): return context.Database.SelectLabs().Result
    task = asyncio.create_task(getLabs())

    # do other stuff...
    await asyncio.sleep(1)

    # ... until you need the data
    labs = await task

    print('All labs:')
    print([lab.Name for lab in labs])

    labOfActiveController = next((l for l in labs
                                  if any (c.Id == context.Database.ActiveController.Id for c in l.ControllerInfos)),
                                 None)

    print(f'Controllers in {labOfActiveController.Name}:')
    print([controller.Name for controller in labOfActiveController.ControllerInfos])

    print(f'Fingerprints in {labOfActiveController.Name}:')
    print([fingerprint.ToString() for controller in labOfActiveController.ControllerInfos for fingerprint in controller.Fingerprints])

    print(context.Logic.CurrentState)

    context.Dispose()

loop = asyncio.get_event_loop()

# The following line is only needed as workaround for a bug in Spyder
# You may skip it in other IDEs e.g. PyCharm or when running from console.
#
# (see https://github.com/spyder-ide/spyder/issues/7096)
nest_asyncio.apply(loop)

loop.run_until_complete(main())
