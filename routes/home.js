import express from "express";
import { spawn } from 'child_process';

const router = express.Router()
//router.get('/', (req, res) =>{
//   res.send('hellolate');
//})

router.get('/', function(req,res){
       const python = spawn('python', ['home.py']);
          python.stdout.on('data', function(data){                
        res.write(data)
        res.end('done')
    })
})

export default router 


