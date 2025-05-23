package com.example;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.web.bind.annotation.*;

@RestController
public class LogController {

    private static final Logger logger = LogManager.getLogger(LogController.class);

    @PostMapping("/log")
    public String logInput(@RequestBody String input) {
        if (input.contains("${jndi:")) {
            return "Invalid input detected";
        }
        logger.info("User input: " + input);
        return "Logged: " + input;
    }
}
