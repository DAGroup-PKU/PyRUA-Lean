// Real footage selected from the corresponding workstations. Full recordings preserve their entire timeline.
export const MEDIA = {
  "real": [
    {
      "id": "tic-tac-toe",
      "title": "Tic-tac-toe",
      "property": "Reasoning \u00b7 long-horizon",
      "poster": "assets/tasks/tic-tac-toe.png",
      "sources": [
        "native8x/tic/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/tic-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/tic/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "five-block-stacking",
      "title": "Five-block stacking",
      "property": "Precision \u00b7 multi-object",
      "poster": "assets/tasks/five-block-stacking.png",
      "sources": [
        "native8x/stack/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/stack-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/stack/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "tower-of-hanoi",
      "title": "Tower of Hanoi",
      "property": "Precision \u00b7 long-horizon",
      "poster": "assets/tasks/tower-of-hanoi.png",
      "sources": [
        "media/hanoi-full-v2/index.m3u8",
        null,
        null
      ],
      "videoPoster": "media/hanoi-full-v2/poster.jpg",
      "trace": null,
      "recordingNote": "Stacking then unstacking \u00b7 8x \u00b7 Full scene",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "towel-folding",
      "title": "Towel folding",
      "property": "Deformable \u00b7 bimanual",
      "poster": "assets/tasks/towel-folding.png",
      "sources": [
        "native8x/towel/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/towel-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/towel/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "push-t",
      "title": "Push T",
      "property": "Contact-rich \u00b7 precision",
      "poster": "assets/tasks/push-t.png",
      "sources": [
        "native8x/push/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/push-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/push/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "block-sorting",
      "title": "Block sorting",
      "property": "Multi-object \u00b7 reasoning",
      "poster": "assets/tasks/block-sorting.png",
      "sources": [
        "native8x/sorting/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/sorting-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/sorting/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "block-storage",
      "title": "Block storage",
      "property": "Multi-object manipulation",
      "poster": "assets/tasks/block-storage.png",
      "sources": [
        "native8x/storage/adaptive.m3u8",
        null,
        null
      ],
      "videoPoster": "media/storage-full-poster.jpg?v=16",
      "trace": null,
      "recordingNote": "Complete recording \u00b7 Original framing",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "originalSources": [
        "media/streams/storage/index.m3u8?v=15",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8
    },
    {
      "id": "ethernet-insertion",
      "title": "Ethernet insertion",
      "property": "Precision \u00b7 online-assisted",
      "poster": "assets/tasks/ethernet-insertion.png",
      "sources": [
        "native8x/ethernet/adaptive.m3u8",
        null,
        null
      ],
      "trace": null,
      "recordingNote": "Leading idle shortened; original frames preserved",
      "cameraNames": [
        "FULL VIEW",
        "LEFT WRIST",
        "RIGHT WRIST"
      ],
      "videoPoster": "media/ethernet-r05-poster.jpg",
      "onlineCoding": true,
      "sourcePage": "https://robot-task-evaluation.quincy-wang.chatgpt.site/?run=astra-ethernet-r05",
      "originalSources": [
        "media/ethernet-r05-full.mp4",
        null,
        null
      ],
      "playbackRate": 1,
      "timelineScale": 8,
      "trimmed": true,
      "trimStartOriginalSeconds": 210,
      "startAt": 1.208269
    }
  ],
  "sim": [
    {
      "id": "classify-objects",
      "title": "Classify objects",
      "property": "",
      "sources": [
        "simulation/classify-objects-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/classify-objects-complete.jpg",
      "videoPoster": "simulation/classify-objects-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "deposit-coin",
      "title": "Deposit coin",
      "property": "",
      "sources": [
        "simulation/deposit-coin-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/deposit-coin-complete.jpg",
      "videoPoster": "simulation/deposit-coin-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "stack-blocks",
      "title": "Stack blocks",
      "property": "",
      "sources": [
        "simulation/stack-blocks-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/stack-blocks-complete.jpg",
      "videoPoster": "simulation/stack-blocks-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "imitate-sorting-sequence",
      "title": "Imitate sorting sequence",
      "property": "",
      "sources": [
        "simulation/imitate-sorting-sequence-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/imitate-sorting-sequence-complete.jpg",
      "videoPoster": "simulation/imitate-sorting-sequence-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "play-stacking-toy",
      "title": "Play stacking toy",
      "property": "",
      "sources": [
        "simulation/play-stacking-toy-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/play-stacking-toy-complete.jpg",
      "videoPoster": "simulation/play-stacking-toy-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "plug-in-charger",
      "title": "Plug in charger",
      "property": "",
      "sources": [
        "simulation/plug-in-charger-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/plug-in-charger-complete.jpg",
      "videoPoster": "simulation/plug-in-charger-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "build-tower",
      "title": "Build tower",
      "property": "",
      "sources": [
        "simulation/build-tower-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/build-tower-complete.jpg",
      "videoPoster": "simulation/build-tower-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    },
    {
      "id": "pour-liquid-into-cup",
      "title": "Pour liquid into cup",
      "property": "",
      "sources": [
        "simulation/pour-liquid-into-cup-complete.mp4",
        null,
        null
      ],
      "trace": null,
      "poster": "simulation/pour-liquid-into-cup-complete.jpg",
      "videoPoster": "simulation/pour-liquid-into-cup-complete.jpg",
      "playbackRate": 1,
      "recordingNote": "Successful complete trajectory",
      "cameraNames": [
        "HEAD",
        "LEFT WRIST",
        "RIGHT WRIST"
      ]
    }
  ]
};
export const CAMERA_NAMES = {"real":["HEAD","LEFT WRIST","RIGHT WRIST"],"sim":["VIEW A","VIEW B","VIEW C"]};
export const CINEMATIC = [
  {
    "id": "stack-1",
    "taskId": "five-block-stacking",
    "title": "Five-block stacking",
    "detail": "Build the tower",
    "speed": 2,
    "src": "media/stack-1.mp4?v=15",
    "poster": "media/stack-1.jpg?v=15"
  },
  {
    "id": "tic-2",
    "taskId": "tic-tac-toe",
    "title": "Tic-tac-toe",
    "detail": "Left wrist \u00b7 Place a piece",
    "speed": 2,
    "src": "media/tic-2.mp4?v=15",
    "poster": "media/tic-2.jpg?v=15"
  },
  {
    "id": "sorting-1",
    "taskId": "block-sorting",
    "title": "Block sorting",
    "detail": "DJI original highlight",
    "speed": 2,
    "src": "media/hero-sorting.mp4?v=15",
    "poster": "media/hero-sorting.jpg?v=15",
    "view": "DJI"
  },
  {
    "id": "ethernet-1",
    "taskId": "ethernet-insertion",
    "title": "Ethernet insertion",
    "detail": "Approach and alignment",
    "speed": 2,
    "src": "media/ethernet-1.mp4?v=15",
    "poster": "media/ethernet-1.jpg?v=15",
    "view": "Left wrist"
  },
  {
    "id": "hanoi-2",
    "taskId": "tower-of-hanoi",
    "title": "Tower of Hanoi",
    "detail": "Left wrist \u00b7 Lift & transfer",
    "speed": 3,
    "src": "media/hanoi-2.mp4?v=15",
    "poster": "media/hanoi-2.jpg?v=15"
  },
  {
    "id": "storage-1",
    "taskId": "block-storage",
    "title": "Block storage",
    "detail": "Object collection",
    "speed": 1.5,
    "src": "media/storage-1.mp4?v=15",
    "poster": "media/storage-1.jpg?v=15",
    "view": "Head"
  },
  {
    "id": "towel-1",
    "taskId": "towel-folding",
    "title": "Towel folding",
    "detail": "Fold",
    "speed": 1.5,
    "src": "media/towel-1.mp4?v=15",
    "poster": "media/towel-1.jpg?v=15"
  },
  {
    "id": "push-2",
    "taskId": "push-t",
    "title": "Push T",
    "detail": "Fine adjustment",
    "speed": 2,
    "src": "media/push-2.mp4?v=15",
    "poster": "media/push-2.jpg?v=15"
  },
  {
    "id": "push-1",
    "taskId": "push-t",
    "title": "Push T",
    "detail": "Coarse alignment",
    "speed": 2,
    "src": "media/push-1.mp4?v=15",
    "poster": "media/push-1.jpg?v=15"
  },
  {
    "id": "ethernet-2",
    "taskId": "ethernet-insertion",
    "title": "Ethernet insertion",
    "detail": "Insertion and release",
    "speed": 2,
    "src": "media/ethernet-2.mp4?v=15",
    "poster": "media/ethernet-2.jpg?v=15",
    "view": "Left wrist"
  },
  {
    "id": "hanoi-1",
    "taskId": "tower-of-hanoi",
    "title": "Tower of Hanoi",
    "detail": "Pick & stack \u00b7 DJI close-up",
    "speed": 3,
    "src": "media/hanoi-1.mp4?v=15",
    "poster": "media/hanoi-1.jpg?v=15"
  },
  {
    "id": "stack-2",
    "taskId": "five-block-stacking",
    "title": "Five-block stacking",
    "detail": "Right wrist \u00b7 Stack a block",
    "speed": 1.5,
    "src": "media/stack-2.mp4?v=15",
    "poster": "media/stack-2.jpg?v=15"
  },
  {
    "id": "towel-2",
    "taskId": "towel-folding",
    "title": "Towel folding",
    "detail": "Unfold",
    "speed": 1.5,
    "src": "media/towel-2.mp4?v=15",
    "poster": "media/towel-2.jpg?v=15"
  },
  {
    "id": "sorting-2",
    "taskId": "block-sorting",
    "title": "Block sorting",
    "detail": "Right wrist \u00b7 Sort to blue",
    "speed": 1.5,
    "src": "media/sorting-2.mp4?v=15",
    "poster": "media/sorting-2.jpg?v=15"
  },
  {
    "id": "storage-2",
    "taskId": "block-storage",
    "title": "Block storage",
    "detail": "Right wrist \u00b7 Place in bowl",
    "speed": 1.5,
    "src": "media/storage-2.mp4?v=15",
    "poster": "media/storage-2.jpg?v=15"
  },
  {
    "id": "tic-1",
    "taskId": "tic-tac-toe",
    "title": "Tic-tac-toe",
    "detail": "Pick & place \u00b7 DJI close-up",
    "speed": 2,
    "src": "media/tic-1.mp4?v=15",
    "poster": "media/tic-1.jpg?v=15"
  }
];

export const HERO_CLIPS = [
  {
    "id": "tic-1",
    "src": "media/tic-1.mp4?v=15",
    "poster": "media/tic-1.jpg?v=15"
  },
  {
    "id": "hanoi-1",
    "src": "media/hanoi-1.mp4?v=15",
    "poster": "media/hanoi-1.jpg?v=15"
  },
  {
    "id": "hero-stack",
    "src": "media/hero-stack.mp4?v=15",
    "poster": "media/hero-stack.jpg?v=15"
  },
  {
    "id": "hero-sorting",
    "src": "media/hero-sorting.mp4?v=15",
    "poster": "media/hero-sorting.jpg?v=15"
  }
];
