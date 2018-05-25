function renderObject(options) {
  const obj = document.createElement(options.element);

  Object.keys(options).forEach(function(attr) {
    // recursive case:
    // appendChild for each child
    if (attr == 'children') {
      options[attr].forEach(function(child) {
        if (child instanceof HTMLElement) {
          obj.appendChild(child);
        } else {
          obj.appendChild(renderObject(child));
        }
      });
    } else if (attr != 'element') {
      // otherwise:
      // fill in attribute
      // for cases like "style.color", recurse and update attribute
      let attrs = attr.split(".");
      let obj_ptr = obj;

      for (let i = 0; i < attrs.length-1; i++) {
        obj_ptr = obj_ptr[attrs[i]];
      }

      obj_ptr[attrs[attrs.length-1]] = options[attr];
    }
  });

  return obj;
}
