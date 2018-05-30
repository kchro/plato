/**
 *
 */
function find(root, selector) {
  return $(root).find(selector)[0];
}

function createCORSRequest(method, url){
    var xhr = new XMLHttpRequest();
    if ("withCredentials" in xhr){
        xhr.open(method, url, true);
    } else if (typeof XDomainRequest != "undefined"){
        xhr = new XDomainRequest();
        xhr.open(method, url);
    } else {
        xhr = null;
    }
    return xhr;
}

/**
 * post
 */
function post(fol, params, callback) {
  let formData = new FormData();
  formData.append("sig", fol);
  formData.append("rules", params.rules);
  formData.append("rulefile", params.rulefile);
  formData.append("blockrules", params.blockrules);
  formData.append("blockfile", params.blockfile);

  let request = createCORSRequest('POST', 'http://cypriot.stanford.edu:8080/ace/');
  request.onreadystatechange = function() {
    if (request.readyState == 4 && request.status == 200) {
      callback(request);
    } else {
      console.log(request);
    }
  }

  console.log(formData);
  request.send(formData);
}

function renderApp(root, state) {
  const app = renderObject({
    element: 'div',
    id: 'app',
    style: {
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      overflowY: 'hidden'
    },
    children: [
      {
        element: 'div',
        id: 'chat-window',
        style: {
          width: '100%',
          minHeight: 'calc(100% - 50px)',
          maxHeight: 'calc(100% - 50px)',
          overflowY: 'auto'
        }
      },

      // Bootstrap Input group (https://getbootstrap.com/docs/4.0/components/input-group/#basic-example)
      {
        element: 'div',
        id: 'action-bar',
        className: 'input-group',
        style: {
          width: '100%',
          height: '50px'
        },
        children: [
          {
            element: 'input',
            id: 'user-input',
            className: 'form-control',
            placeholder: 'Enter response here'
          },
          {
            element: 'div',
            className: 'input-group-append',
            children: [
              {
                element: 'button',
                id: 'new-problem',
                className: 'btn btn-outline-secondary',
                type: 'button',
                innerHTML: 'New Problem' // maybe this is just an action button
              },
              {
                element: 'button',
                id: 'send',
                className: 'btn btn-outline-secondary',
                type: 'button',
                innerHTML: 'Send'
              }
            ]
          }
        ]
      }
    ]
  });

  return app;
}

function startApp(root, state) {
  const dialog = [
    'Hello, my name is Plato.',
    'Let\'s do some translation exercises. Please press the \'New Problem\' button to get started.',
  ]

  const chatWindow = document.getElementById('chat-window');
  chatWindow.appendChild(renderMessage({
    'message': dialog[0]
  }));
  chatWindow.appendChild(renderMessage({
    'displayIcon': true,
    'message': dialog[1]
  }));

  $('#new-problem').on('click', function(e) {
    chatWindow.appendChild(renderMessage({
      user: true,
      message: 'New Problem!'
    }));

    let message = renderTyping();
    chatWindow.appendChild(message);
    post(data.problems[state.problem].answer, data.config, function(response) {
      // get the list of NL sentences and choose based on policy
      const prompts = response.responseText.split('\n');

      // random policy
      const idx = Math.floor(Math.random() * prompts.length);

      clearTimeout(message.interval);
      message.update('Please translate this sentence into First-Order Logic (FOL):');
      message.display(false);
      chatWindow.appendChild(renderMessage({
        'displayIcon': true,
        'message': prompts[idx]
      }))
    });
  });

  $('#send').on('click', function(e) {
    const userInput = document.getElementById('user-input');
    if (userInput.value.length > 0) {
      chatWindow.appendChild(renderMessage({
        user: true,
        message: userInput.value
      }));

      const link = getEndpointLink(userInput.value);

      alert(link);

      userInput.value = '';
    }
  });
}

/**
 *
 */
function render(root, state) {
  let app, body;

  if (state.init == false) {
    /*
     * render the application structure
     */
    app = renderApp(root, state);
    body = $(root).find(".card-body")[0];
    body.appendChild(app);

    // structure is built
    state.init = true;

    // begin dialog:
    startApp(root, state);

  } else if (state.problem >= data.problems.length) {
    // TODO: show analytics
  } else {
    // rerender the app
    body = $(root).find(".card-body")[0];
    $(body).empty();

    app = renderApp(root, state);
    body.appendChild(app);
  }
  console.log(root);
}

/**
 *
 */
$(function() {
  let root = document.getElementById("plato-app");

  const state = {
    "init": false,
    "problem": 2,
    "random": true,
    "analytics": [
      {
        "label":"correct",
        "value":0
      },
      {
        "label":"incorrect",
        "value":0
      }
    ],
    "error_log": {}
  };

  render(root, state);
});
