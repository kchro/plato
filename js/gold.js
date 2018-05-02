data = {
  "name":"plato",
  "problems":[
    {
      "label":"Problem 0.1",
      "prompt":"a is a cube, and b is a cube.",
      "answer":"cube(a) & cube(b)"
    },
    {
      "label":"Problem 0.2",
      "prompt":"a and b are cubes.",
      "answer":"cube(a) & cube(b)"
    },
    {
      "label":"Problem 0.3",
      "prompt":"a is a small cube.",
      "answer":"small(a) & cube(a)"
    },
    {
      "label":"Problem 0.4",
      "prompt":"a and b are small cubes",
      "answer":"small(a) & cube(a) & small(b) & cube(b)"
    },
    {
      "label":"Problem 0.5",
      "prompt":"all cubes are small",
      "answer":"all_x(cube(x) -> small(x))"
    }
  ]
};
