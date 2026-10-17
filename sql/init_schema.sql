CREATE DATABASE IF NOT EXISTS smart_mlops;
USE smart_mlops;

CREATE TABLE IF NOT EXISTS models (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    domain VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id INT AUTO_INCREMENT PRIMARY KEY,
    model_id INT NOT NULL,
    version_tag VARCHAR(20) NOT NULL,
    accuracy DECIMAL(5,4) NOT NULL,
    f1_score DECIMAL(5,4) NOT NULL,
    created_by VARCHAR(50) NOT NULL,
    is_active_prod BOOLEAN NOT NULL DEFAULT FALSE,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_versions_model FOREIGN KEY (model_id) REFERENCES models(model_id),
    CONSTRAINT uq_model_version UNIQUE (model_id, version_tag)
);

CREATE TABLE IF NOT EXISTS release_history (
    release_id INT AUTO_INCREMENT PRIMARY KEY,
    version_id INT NOT NULL,
    promoted_by VARCHAR(50) NOT NULL,
    promotion_status ENUM('SUCCESS','ROLLED_BACK') NOT NULL,
    reason VARCHAR(255),
    promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_history_version FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
);

DELIMITER $$

DROP PROCEDURE IF EXISTS PromoteModel$$

CREATE PROCEDURE PromoteModel(
    IN p_model_name VARCHAR(100),
    IN p_version_tag VARCHAR(20),
    IN p_accuracy DECIMAL(5,4),
    IN p_f1_score DECIMAL(5,4),
    IN p_developer VARCHAR(50)
)
BEGIN
    DECLARE v_model_id INT;
    DECLARE v_current_prod_acc DECIMAL(5,4) DEFAULT NULL;
    DECLARE v_new_version_id INT;

    START TRANSACTION;

    SELECT model_id INTO v_model_id
    FROM models
    WHERE model_name = p_model_name
    LIMIT 1;

    IF v_model_id IS NULL THEN
        INSERT INTO models (model_name, domain)
        VALUES (p_model_name, 'German Credit Risk');
        SET v_model_id = LAST_INSERT_ID();
    END IF;

    SELECT accuracy INTO v_current_prod_acc
    FROM model_versions
    WHERE model_id = v_model_id AND is_active_prod = TRUE
    LIMIT 1 FOR UPDATE;

    IF v_current_prod_acc IS NOT NULL AND p_accuracy <= v_current_prod_acc THEN

        INSERT INTO model_versions
            (model_id, version_tag, accuracy, f1_score, created_by, is_active_prod)
        VALUES
            (v_model_id, p_version_tag, p_accuracy, p_f1_score, p_developer, FALSE);
        SET v_new_version_id = LAST_INSERT_ID();

        INSERT INTO release_history
            (version_id, promoted_by, promotion_status, reason)
        VALUES
            (v_new_version_id, p_developer, 'ROLLED_BACK',
             CONCAT('Rejected: Candidate accuracy (', p_accuracy,
                    ') <= current production (', v_current_prod_acc, ')'));

        ROLLBACK;

        SELECT model_id INTO v_model_id
        FROM models
        WHERE model_name = p_model_name
        LIMIT 1;

        IF v_model_id IS NULL THEN
            INSERT INTO models (model_name, domain)
            VALUES (p_model_name, 'German Credit Risk');
            SET v_model_id = LAST_INSERT_ID();
        END IF;

        INSERT INTO model_versions
            (model_id, version_tag, accuracy, f1_score, created_by, is_active_prod)
        VALUES
            (v_model_id, p_version_tag, p_accuracy, p_f1_score, p_developer, FALSE);
        SET v_new_version_id = LAST_INSERT_ID();

        INSERT INTO release_history
            (version_id, promoted_by, promotion_status, reason)
        VALUES
            (v_new_version_id, p_developer, 'ROLLED_BACK',
             CONCAT('Rejected: Candidate accuracy (', p_accuracy,
                    ') <= current production (', v_current_prod_acc, ')'));

        COMMIT;

        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Gatekeeper Alert: Candidate accuracy is not strictly higher than the current production baseline.';

    ELSE

        UPDATE model_versions
        SET is_active_prod = FALSE
        WHERE model_id = v_model_id AND is_active_prod = TRUE;

        INSERT INTO model_versions
            (model_id, version_tag, accuracy, f1_score, created_by, is_active_prod)
        VALUES
            (v_model_id, p_version_tag, p_accuracy, p_f1_score, p_developer, TRUE);
        SET v_new_version_id = LAST_INSERT_ID();

        INSERT INTO release_history
            (version_id, promoted_by, promotion_status, reason)
        VALUES
            (v_new_version_id, p_developer, 'SUCCESS', 'Passed quality gate check.');

        COMMIT;

    END IF;
END$$

DELIMITER ;

CALL PromoteModel('German_Credit_Classifier', 'v1.0', 0.7500, 0.7400, 'system_init');
