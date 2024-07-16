# -*- coding: utf-8 -*-
"""
Please configure 'dll', 'db' and 'vm' before running any of the examples.

dll: Path to the SDK's DLLs that usually reside in neaSCAN's application folder

db: Address of the database, which is either
    - A centralized database server (e.g. 10.99.10.50), or
    - Embedded database on neaSNOM (e.g. 10.82.77.1)

vm: Fingerprint of the neaSNOM, which can be obtained from

    - neaSCAN logfile:
      Upon connection you will find a line like this:
        "Server Fingerprint: [00313679-3dbe-4d4a-9105-16835504fc97] has authenticated"

    - Service Tools:
      Run this line in the Python Console:
        print([id.ToString() for id in context.Database.ActiveController.Fingerprints])

    - neaSNOM:
      Upon startup you will find in the logfile a line similar to this:
        "Server Version: 1.10.6486+, Fingerprint: [00313679-3dbe-4d4a-9105-16835504fc97]"

      You can also run this in a terminal in order retrieve the fingerprint at any time:
        cat /etc/neaspec/uuid
"""

dll = r'..\..\bin\Debug'

#db = 'localhost'
db = '10.99.10.50'

#vm = '00313679-3dbe-4d4a-9105-16835504fc97' # 32
vm = '76203d64-d94e-48ab-8df5-4ec614881289' # 64

# vm2 = '2203f348-8420-4fcd-beca-50456f15cd93'
