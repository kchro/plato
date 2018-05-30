String.prototype.replaceAll = function(target, replacement) { return this.split(target).join(replacement); };

String.prototype.hexEncode = function() {
	let hex = "";
	for (var i = 0; i < this.length; i++) {
		hex += "" + this.charCodeAt(i).toString(16);
	}
	return hex;
}

function getEndpointLink(formula) {
  const target = "@#$%^&~|/";
  const replacement = "\u2200\u2260\u2192\u2194\u22A5\u2227\u00AC\u2228\u2203";
  const hexFormula = formula.replaceAll(target, replacement).hexEncode() + "?binary=true";
  const link = "https://fol.gradegrinder.net/rest/FormulaParser/parse/" + hexFormula;
  return link;
}

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
    } else if (attr == 'style') {
      Object.keys(options[attr]).forEach(function(style_attr) {
        obj[attr][style_attr] = options[attr][style_attr];
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

function renderMessage(options) {
  if (options.user) {
    const userMessage = renderObject({
      element: 'div',
      style: {
        display: 'flex',
        justifyContent: 'flex-end',
        margin: '10px 0',
        width: '100%'
      },
      children: [
        {
          element: 'div',
          innerHTML: options.message,
          style: {
            display: 'flex',
            alignItems: 'center',
            borderRadius: '10px',
            background: '#3f91f2',
            padding: '10px',
            color: 'white'
          }
        }
      ]
    });

    return userMessage;
  } else {
    const iconSize = 50;
    const url = 'url("../assets/plato.jpg")';
    const platoMessage = renderObject({
      element: 'div',
      style: {
        display: 'flex',
        margin: '10px 0',
        width: '100%'
      },
      children: [
        {
          'element': 'div',
          'style.borderRadius': iconSize+'px',
          'style.width': iconSize+'px',
          'style.height': iconSize+'px',
          'style.backgroundImage': url,
          'style.backgroundSize': 'contain'
        },
        {
          'element': 'div',
          'style.display': 'flex',
          'style.alignItems': 'center',
          'style.borderRadius': '10px',
          'style.background': '#d7d7d7',
          'style.padding': '10px',
          'style.marginLeft': '20px',
          'innerHTML': options.message
        }
      ]
    });

    platoMessage.update = function(str) {
      platoMessage.children[1].innerHTML = str;
    }

    platoMessage.display = function(show) {
      platoMessage.children[0].style.backgroundImage = (show) ? url : 'none';
    }

    platoMessage.display(options.displayIcon);

    return platoMessage;
  }
}

function renderTyping(callback) {
  const loading = ['.  ', '.. ', '...'];
  let idx = 0;

  const typing = renderMessage({
    displayIcon: true,
    message: loading[idx++]
  });

  typing.style.whiteSpace = 'pre-wrap';

  typing.interval = setInterval(function() {
    typing.update(loading[idx++ % 3]);
  }, 500);

  return typing;
}
