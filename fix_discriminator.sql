-- =====================================================
-- FIX: Add Discriminator column to Users table
-- Run this in Neon SQL Editor
-- =====================================================

-- Add Discriminator column to Users table
ALTER TABLE "Users" ADD COLUMN IF NOT EXISTS "Discriminator" VARCHAR(50) NOT NULL DEFAULT 'User';

-- Update existing rows to have proper discriminator values
-- Users with entries in Admins table should be 'Admin'
UPDATE "Users" u
SET "Discriminator" = 'Admin'
WHERE EXISTS (SELECT 1 FROM "Admins" a WHERE a."Id" = u."Id");

-- Verify the fix
SELECT "Id", "Email", "Role", "Discriminator" 
FROM "Users" 
LIMIT 10;

-- Check if any rows need fixing
SELECT COUNT(*) as total_users,
       COUNT(CASE WHEN "Discriminator" = 'User' THEN 1 END) as users,
       COUNT(CASE WHEN "Discriminator" = 'Admin' THEN 1 END) as admins
FROM "Users";

COMMIT;
