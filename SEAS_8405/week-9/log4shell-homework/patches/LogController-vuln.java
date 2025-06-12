package com.example;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.web.bind.annotation.*;

/**
 * WARNING: This version is intentionally vulnerable to demonstrate Log4Shell.
 */
@RestController
@RequestMapping("/api")
public class LogController {

    private static final Logger logger = LogManager.getLogger(LogController.class);

    @PostMapping("/log")
    public String logInput(@RequestBody String input) {
        // UNSAFE: Directly logs input without validation
        logger.info(input);
        return "Logged: " + input;
    }
}
