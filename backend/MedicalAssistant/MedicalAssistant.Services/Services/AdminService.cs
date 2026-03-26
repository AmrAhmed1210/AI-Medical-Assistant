using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;


namespace MedicalAssistant.Services.Services;

public class AdminService : IAdminService
{
    private readonly IUnitOfWork _unitOfWork;

    public AdminService(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    // ===============================
    // 📊 Stats
    // ===============================
    public async Task<SystemStatsDto> GetSystemStatsAsync()
    {
        var usersCount = await _unitOfWork.Repository<User>().CountAsync();
        var doctorsCount = await _unitOfWork.Repository<Doctor>().CountAsync();
        var patientsCount = await _unitOfWork.Repository<Patient>().CountAsync();

        return new SystemStatsDto
        {
            TotalUsers = usersCount,
            TotalDoctors = doctorsCount,
            TotalPatients = patientsCount,
            TotalAppointments = 0
        };
    }

    // ===============================
    // 👥 Get Users (Pagination + Filter)
    // ===============================
    public async Task<object> GetUsersAsync(
        int page = 1,
        int pageSize = 10,
        string? search = null,
        string? role = null)
    {
        var users = await _unitOfWork.Repository<User>().GetAllAsync();

        var query = users.AsQueryable();

        // 🔍 search
        if (!string.IsNullOrWhiteSpace(search))
        {
            search = search.ToLower();
            query = query.Where(u =>
                u.FullName.ToLower().Contains(search) ||
                u.Email.ToLower().Contains(search));
        }

        // 🎯 role filter
        if (!string.IsNullOrWhiteSpace(role))
        {
            query = query.Where(u => u.Role == role);
        }

        var total = query.Count();

        var data = query
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .Select(u => new
            {
                id = u.Id,
                name = u.FullName,
                email = u.Email,
                role = u.Role,
                isActive = u.IsActive
            })
            .ToList();

        return new
        {
            items = data,
            total = total
        };
    }

    // ===============================
    // 🔄 Toggle
    // ===============================
    public async Task<bool> ToggleUserStatusAsync(int userId)
    {
        var repo = _unitOfWork.Repository<User>();
        var user = await repo.GetByIdAsync(userId);

        if (user == null) return false;

        user.IsActive = !user.IsActive;

        await _unitOfWork.SaveChangesAsync();
        return true;
    }

    // ===============================
    // ❌ Delete
    // ===============================
    public async Task<bool> DeleteUserAsync(int id)
    {
        var repo = _unitOfWork.Repository<User>();
        var user = await repo.GetByIdAsync(id);

        if (user == null) return false;

        repo.Delete(user);
        await _unitOfWork.SaveChangesAsync();

        return true;
    }

    // ===============================
    // ➕ Create
    // ===============================
    public async Task<object> CreateUserAsync(CreateUserRequest request)
    {
        var user = new User
        {
            FullName = request.FullName,
            PasswordHash = request.PasswordHash,
            Email = request.Email,
            Role = request.Role,
            IsActive = true
        };

        await _unitOfWork.Repository<User>().AddAsync(user);
        await _unitOfWork.SaveChangesAsync();

        return new
        {
            id = user.Id,
            name = user.FullName,
            passwordHash = user.PasswordHash,
            email = user.Email,
            role = user.Role,
            isActive = user.IsActive
        };
    }
}