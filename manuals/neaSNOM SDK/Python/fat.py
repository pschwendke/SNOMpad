import asyncio

import neaspec

# Instead of awaiting the task, it is also possible to use the property "Result",
# which is blocking until the task completes. 
#
# print(neaspec.context.Microscope.GetCalibrationSetting("regulator-pid").Result)

async def main():

	# This script can be run in both ways:
	# 1. Within neaSCAN reusing the given context, or
	# 2. Standalone: Connecting and creating the context
	if neaspec.context is None:
		print("Waiting for connection...");
		
		import configuration as config
		await neaspec.connect(config.db, config.vm, config.dll)

	async def getCalibration(key: str):
		return neaspec.context.Microscope.GetCalibrationSetting(key).Result
		
	async def setCalibration(key: str, value: str):
		return neaspec.context.Microscope.SetCalibrationSetting("regulator-pid", value).Result

	print("Regulator PID old value:")	
	old_value = await asyncio.create_task(getCalibration("regulator-pid"))
	print(old_value)

	await asyncio.create_task(setCalibration("regulator-pid", "[1.0, 2.0, 3.0]"))
	
	print("Regulator PID new value:")
	new_value = await asyncio.create_task(getCalibration("regulator-pid"))
	print(new_value)

	# Revert changes
	await asyncio.create_task(setCalibration("regulator-pid", old_value))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())