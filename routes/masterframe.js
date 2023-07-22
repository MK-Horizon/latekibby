import express from 'express'
import { spawn } from 'child_process'

const router  = express.Router()



{/* starting with bitcoin all data*/}
router.get('/technical', (req, res) => {
    const python = spawn('python', ['6hpricedatagen.py'])
    python.stdout.on('data', function(data){
        console.log(data.toString())
        res.write(data)
        res.end('end')
        
    })
       
})

router.get('/readable', (req, res) => {
    const python = spawn('python', ['6hpricedatahuman.py'])
    python.stdout.on('data', function(data){
        console.log(data.toString())
        res.write(data)
        res.end('end')
        
    })
       
})

router.get('/', (req, res) => {
    const python = spawn('python', ['6hpricedata.py'])
    python.stdout.on('data', function(data){
        console.log(data.toString())
        res.write(data)  
    })  
})

export default router;