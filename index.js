import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

import usersRoutes from './routes/users.js';
import masterframe from './routes/masterframe.js';
import home from './routes/home.js';

const app = express();
const PORT = 7000;

app.use(cors({ origin: '*' }));
app.use(bodyParser.json());

app.use('/users', usersRoutes);
app.use('/bitcoin/ohlcpricedata/6hour', masterframe);
app.use('/', home);

// Error handling middleware for 404 Not Found
app.use((req, res, next) => {
  const error = new Error('Not Found');
  error.status = 404;
  next(error);
});

// Error handling middleware for other errors
app.use((err, req, res, next) => {
  res.status(err.status || 500);
  res.json({
    error: {
      message: err.message,
    },
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
