import express from 'express'
const router = express.Router()

{/* starting with users*/}
router.get('/', (req, res) => {res.send('hello')})

export default router;